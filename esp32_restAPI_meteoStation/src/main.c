#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>

#include "driver/adc.h"
#include "esp_adc_cal.h"
#include "driver/gpio.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "rom/ets_sys.h"
#include <driver/i2c.h>
#include <math.h>
#include "sht21.h"
#include <i2cdev.h>
#include <ms5611.h>
#include <inttypes.h>

#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_netif.h"

#include "esp_http_client.h"
#include "mbedtls/md.h"
#include "mbedtls/base64.h"
#include "esp_mac.h"

#define I2C_MASTER_SDA_IO   21
#define I2C_MASTER_SCL_IO   22
#define I2C_MASTER_PORT     I2C_NUM_0
#define I2C_MASTER_FREQ_HZ  100000

static const char *WIFI_SSID   = "METEO_AP";
static const char *WIFI_PASS   = "meteo_jetson_nano_passwd";

static const char *JWT_SECRET  = "V3fryS3cr3tK3y,7h47Y0uC4n7F1nd:)!";

// HTTPS URLs – keep only if your Jetson really speaks TLS on 5500
static const char *SEND_DATA_URL      = "https://192.168.50.1:5500/get_data";
static const char *GENERATE_TOKEN_URL = "https://192.168.50.1:5500/generate_token";
static const char *REQUEST_ACCESS_URL = "https://192.168.50.1:5500/request";

#define POST_PERIOD_MS 5000
static const char *TAG = "main";

// ------------ FSM ------------

typedef enum {
    SEND_STATE = 0,
    GENERATE_TOKEN_STATE,
    REQUEST_ACCESS_STATE
} current_state_t;

// ------------ meteo sample ------------

typedef struct {
    float temperature;
    float humidity;
    float wind_speed;
    float wind_direction_degrees;
    float rainfall;
    float dust;
    float pressure;
    float altitude;
    uint32_t timestamp_ms;
} meteo_sample_t;

static meteo_sample_t g_sample;
static SemaphoreHandle_t g_sample_mutex;

static current_state_t CURRENT_STATE = SEND_STATE;
static const char *current_url = NULL;

// JWT header pointer
static char *header_jwt = NULL;

// Sensor availability flags
static bool g_ms5611_available = false;

// ------------ sensors / GPIO ------------

void sht21_task(void *pvParameters);
void ms5611_task(void *pvParameters);
void gp2y_task(void *pvParameters);
static ms5611_t ms_dev;

const gpio_num_t rain_pin       = GPIO_NUM_33;
const gpio_num_t wind_speed_pin = GPIO_NUM_32;

#define LED_PIN                    4
#define GP2Y_ADC_CHANNEL           ADC1_CHANNEL_6 
#define WIND_DIRECTION_ADC_CHANNEL ADC1_CHANNEL_7 

#define DEFAULT_VREF   1100
#define NO_OF_SAMPLES  64
#define VCC_SENSOR     5.0f
#define ADC_FULL_SCALE 3.3f

static esp_adc_cal_characteristics_t *adc_chars = NULL;

#define MM_PER_TIP       0.2794f
#define KPH_PER_HZ       2.4f

#define WIND_DEBOUNCE_US   (500)
#define RAIN_DEBOUNCE_US   (100000LL)

typedef struct {
    int64_t timestamp_us;
} sensor_event_t;

#define WMK_NUM_ANGLES    16
#define WMK_DEG_PER_INDEX 22.5f

static uint16_t vaneADCValues[WMK_NUM_ANGLES] = {
    126, 126, 337, 337, 559, 559, 1196, 1196,
    1630, 1630, 1569, 1569, 1402, 1402, 887, 887
};

#define WIND_DIR_OFFSET_DEG  0.0f

static QueueHandle_t rain_queue       = NULL;
static QueueHandle_t wind_speed_queue = NULL;

// ------------ helpers ------------

static float calculate_altitude_m(int32_t pressure_pa, float sea_level_hpa)
{
    float pressure_hpa = pressure_pa / 100.0f;
    return 44330.0f * (1.0f - powf(pressure_hpa / sea_level_hpa, 1.0f / 5.255f));
}

static float normalize_angle(float deg)
{
    while (deg < 0.0f)    deg += 360.0f;
    while (deg >= 360.0f) deg -= 360.0f;
    return deg;
}

static const char* cardinal_from_angle(float deg)
{
    deg = normalize_angle(deg);
    if (deg < 22.5f || deg >= 337.5f) return "N";
    if (deg < 67.5f)  return "NE";
    if (deg < 112.5f) return "E";
    if (deg < 157.5f) return "SE";
    if (deg < 202.5f) return "S";
    if (deg < 247.5f) return "SW";
    if (deg < 292.5f) return "W";
    return "NW";
}

static void getWindDirectionFromRaw(uint16_t rawADC, float *angle_deg, const char **cardinal)
{
    uint16_t closestIndex = 0;
    uint16_t lastDiff     = 0xFFFF;

    for (uint8_t i = 0; i < WMK_NUM_ANGLES; i++) {
        uint16_t base = vaneADCValues[i];
        uint16_t diff = (rawADC > base) ? (rawADC - base) : (base - rawADC);
        if (diff < lastDiff) {
            lastDiff = diff;
            closestIndex = i;
        }
    }

    float base_deg  = closestIndex * WMK_DEG_PER_INDEX;
    float corrected = normalize_angle(base_deg + WIND_DIR_OFFSET_DEG);

    *angle_deg = corrected;
    *cardinal  = cardinal_from_angle(corrected);
}

static void init_adc(void)
{
    ESP_ERROR_CHECK(adc1_config_width(ADC_WIDTH_BIT_12));
    ESP_ERROR_CHECK(adc1_config_channel_atten(GP2Y_ADC_CHANNEL,        ADC_ATTEN_DB_11));
    ESP_ERROR_CHECK(adc1_config_channel_atten(WIND_DIRECTION_ADC_CHANNEL, ADC_ATTEN_DB_11));

    adc_chars = calloc(1, sizeof(esp_adc_cal_characteristics_t));
    if (!adc_chars) {
        ESP_LOGE(TAG, "Failed to allocate adc_chars");
        abort();
    }

    esp_adc_cal_value_t vt =
        esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11,
                                 ADC_WIDTH_BIT_12, DEFAULT_VREF, adc_chars);
    ESP_LOGI(TAG, "ADC cal type: %d", vt);
}

static void init_led(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask   = (1ULL << LED_PIN),
        .mode           = GPIO_MODE_OUTPUT,
        .pull_up_en     = GPIO_PULLUP_DISABLE,
        .pull_down_en   = GPIO_PULLDOWN_DISABLE,
        .intr_type      = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf);
    gpio_set_level(LED_PIN, 0);
}

static float gp2y_read_voltage(void)
{
    uint32_t adc_reading = 0;

    gpio_set_level(LED_PIN, 1);
    ets_delay_us(280);

    for (int i = 0; i < NO_OF_SAMPLES; i++) {
        adc_reading += adc1_get_raw(GP2Y_ADC_CHANNEL);
    }
    adc_reading /= NO_OF_SAMPLES;

    gpio_set_level(LED_PIN, 0);

    uint32_t mv = esp_adc_cal_raw_to_voltage(adc_reading, adc_chars);
    return mv / 1000.0f;
}

static float gp2y_voltage_to_density_ugm3(float vo)
{
    float density_mg_m3 = 0.170f * vo - 0.1f;
    if (density_mg_m3 < 0.0f) density_mg_m3 = 0.0f;
    return density_mg_m3 * 1000.0f;
}

static uint16_t wind_direction_read_raw(void)
{
    uint32_t raw = 0;
    for (int i = 0; i < NO_OF_SAMPLES; i++) {
        raw += adc1_get_raw(WIND_DIRECTION_ADC_CHANNEL);
    }
    raw /= NO_OF_SAMPLES;
    return (uint16_t)raw;
}

// ------------ ISRs ------------

static void IRAM_ATTR rain_isr_handler(void *arg)
{
    sensor_event_t evt = { .timestamp_us = esp_timer_get_time() };
    BaseType_t hpw = pdFALSE;
    xQueueSendFromISR(rain_queue, &evt, &hpw);
    if (hpw) portYIELD_FROM_ISR();
}

static void IRAM_ATTR wind_speed_isr_handler(void *arg)
{
    sensor_event_t evt = { .timestamp_us = esp_timer_get_time() };
    BaseType_t hpw = pdFALSE;
    xQueueSendFromISR(wind_speed_queue, &evt, &hpw);
    if (hpw) portYIELD_FROM_ISR();
}

static bool isr_inited = false;

static void init_isr(void)
{
    if (isr_inited) return;
    isr_inited = true;

    gpio_config_t io_conf = {
        .intr_type    = GPIO_INTR_NEGEDGE,
        .mode         = GPIO_MODE_INPUT,
        .pin_bit_mask = (1ULL << rain_pin) | (1ULL << wind_speed_pin),
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io_conf));

    ESP_ERROR_CHECK(gpio_install_isr_service(0));
    ESP_ERROR_CHECK(gpio_isr_handler_add(rain_pin,       rain_isr_handler,       (void *)rain_pin));
    ESP_ERROR_CHECK(gpio_isr_handler_add(wind_speed_pin, wind_speed_isr_handler, (void *)wind_speed_pin));
}

// ------------ Wi-Fi / HTTP helpers ------------

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
        ESP_LOGW(TAG, "WiFi disconnected, reconnecting");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

static void wifi_init_sta(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = { 0 };
    strncpy((char*)wifi_config.sta.ssid, WIFI_SSID, sizeof(wifi_config.sta.ssid));
    strncpy((char*)wifi_config.sta.password, WIFI_PASS, sizeof(wifi_config.sta.password));
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
}

static void url_encode(const char *src, char *dst, size_t dst_len)
{
    const char *hex = "0123456789ABCDEF";
    size_t di = 0;

    for (size_t i = 0; src[i] != '\0' && di + 4 < dst_len; ++i) {
        unsigned char c = (unsigned char)src[i];
        if ((c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') ||
            c == '-' || c == '_' || c == '.' || c == '~') {
            dst[di++] = c;
        } else {
            dst[di++] = '%';
            dst[di++] = hex[c >> 4];
            dst[di++] = hex[c & 0x0F];
        }
    }
    dst[di] = '\0';
}

static esp_err_t base64url_from_bin(const unsigned char *in, size_t in_len,
                                    char *out, size_t out_size)
{
    size_t b64_len = 0;
    int ret = mbedtls_base64_encode(NULL, 0, &b64_len, in, in_len);
    if (ret != MBEDTLS_ERR_BASE64_BUFFER_TOO_SMALL) return ESP_FAIL;

    unsigned char *tmp = malloc(b64_len);
    if (!tmp) return ESP_ERR_NO_MEM;

    ret = mbedtls_base64_encode(tmp, b64_len, &b64_len, in, in_len);
    if (ret != 0) { free(tmp); return ESP_FAIL; }

    size_t oi = 0;
    for (size_t i = 0; i < b64_len && oi + 1 < out_size; ++i) {
        char c = (char)tmp[i];
        if (c == '+') c = '-';
        else if (c == '/') c = '_';
        else if (c == '=') { break; }

        out[oi++] = c;
    }
    out[oi] = '\0';
    free(tmp);
    return ESP_OK;
}

static esp_err_t base64url_from_str(const char *in, char *out, size_t out_size)
{
    return base64url_from_bin((const unsigned char*)in, strlen(in), out, out_size);
}

static void get_chip_id_str(char *out, size_t out_len)
{
    uint8_t mac[6];
    esp_err_t err = esp_efuse_mac_get_default(mac);
    if (err != ESP_OK || out_len < 21) {
        if (out_len) out[0] = '\0';
        return;
    }

    uint64_t chip = 0;
    for (int i = 0; i < 6; ++i) {
        chip = (chip << 8) | mac[i];
    }
    snprintf(out, out_len, "%llu", (unsigned long long)chip);
}

static esp_err_t generate_jwt(char *out, size_t out_size)
{
    const char *header_json = "{\"alg\":\"HS256\",\"typ\":\"JWT\"}";
    char payload_json[160];

    char chip_id[32];
    get_chip_id_str(chip_id, sizeof(chip_id));

    snprintf(payload_json, sizeof(payload_json),
             "{\"chip_id\":\"%s\",\"valability\":\"2099-12-31T23:59:59.000000+00:00\"}",
             chip_id);

    char header_b64[128];
    char payload_b64[256];

    ESP_ERROR_CHECK(base64url_from_str(header_json,  header_b64,  sizeof(header_b64)));
    ESP_ERROR_CHECK(base64url_from_str(payload_json, payload_b64, sizeof(payload_b64)));

    char message[512];
    snprintf(message, sizeof(message), "%s.%s", header_b64, payload_b64);

    unsigned char hmac[32];
    mbedtls_md_context_t ctx;
    const mbedtls_md_info_t *info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);

    mbedtls_md_init(&ctx);
    ESP_ERROR_CHECK(mbedtls_md_setup(&ctx, info, 1));
    ESP_ERROR_CHECK(mbedtls_md_hmac_starts(&ctx,
                                           (const unsigned char*)JWT_SECRET,
                                           strlen(JWT_SECRET)));
    ESP_ERROR_CHECK(mbedtls_md_hmac_update(&ctx,
                                           (const unsigned char*)message,
                                           strlen(message)));
    ESP_ERROR_CHECK(mbedtls_md_hmac_finish(&ctx, hmac));
    mbedtls_md_free(&ctx);

    char sig_b64[128];
    ESP_ERROR_CHECK(base64url_from_bin(hmac, sizeof(hmac), sig_b64, sizeof(sig_b64)));

    snprintf(out, out_size, "%s.%s", message, sig_b64);
    return ESP_OK;
}

static int send_segment(const char *base_url,
                        const char *jwt,
                        char **response_out, const meteo_sample_t *sample)
{
    if (response_out) {
        *response_out = NULL;
    }

    if (!base_url || !jwt || !sample) {
        ESP_LOGE(TAG, "send_segment: invalid args");
        return -1;
    }

    char jwt_encoded[1024];
    url_encode(jwt, jwt_encoded, sizeof(jwt_encoded));

    char url[512];
    snprintf(url, sizeof(url), "%s?jwt=%s", base_url, jwt_encoded);

    char json[512];
    snprintf(json, sizeof(json),
            "{"
            "\"temperature\":%.2f,"
            "\"humidity\":%.2f,"
            "\"wind_speed\":%.2f,"
            "\"rainfall\":%.1f,"
            "\"wind_direction_degrees\":%.2f,"
            "\"dust\":%.2f,"
            "\"pressure\":%.2f,"
            "\"altitude\":%.2f,"
            "\"ssid\":\"%s\""
            "}",
            sample->temperature,
            sample->humidity,
            sample->wind_speed,
            sample->rainfall,
            sample->wind_direction_degrees,
            sample->dust,
            sample->pressure,
            sample->altitude,
            WIFI_SSID);

    esp_http_client_config_t cfg = {
        .url = url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 10000,
        .transport_type = HTTP_TRANSPORT_OVER_SSL,
        .skip_cert_common_name_check = true,
        .crt_bundle_attach = NULL,
    };

    esp_http_client_handle_t client = esp_http_client_init(&cfg);
    if (!client) {
        ESP_LOGE(TAG, "Failed to init http client");
        return -1;
    }

    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, json, strlen(json));

    esp_err_t err = esp_http_client_perform(client);
    int code = -1;

    if (err == ESP_OK) {
        code = esp_http_client_get_status_code(client);
        int len = esp_http_client_get_content_length(client);

        ESP_LOGI(TAG, "HTTP POST status = %d, content_length = %d", code, len);

        if (len > 0 && response_out) {
            char *buf = malloc(len + 1);
            if (buf) {
                int read = esp_http_client_read_response(client, buf, len);
                if (read >= 0) {
                    buf[read] = '\0';
                    *response_out = buf;
                } else {
                    free(buf);
                }
            }
        }
    } else {
        ESP_LOGE(TAG, "HTTP POST error: %s", esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
    return code;
}

// ------------ sensor tasks ------------

static void wind_direction_task(void *pvParameters)
{
    (void) pvParameters;

    for (;;) {
        uint16_t raw = wind_direction_read_raw();
        float angle_deg;
        const char *card;

        getWindDirectionFromRaw(raw, &angle_deg, &card);

        if (xSemaphoreTake(g_sample_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            g_sample.wind_direction_degrees = angle_deg;
            xSemaphoreGive(g_sample_mutex);
        }

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

static void rain_task(void *pvParameters)
{
    (void) pvParameters;

    sensor_event_t evt;
    int64_t last_rain_us = 0;
    uint32_t tips_total  = 0;
    uint32_t tips_window = 0;

    const float window_seconds = 1.0f;
    const TickType_t period    = pdMS_TO_TICKS((int)(window_seconds * 1000));
    TickType_t last_wake       = xTaskGetTickCount();

    for (;;) {
        while (xQueueReceive(rain_queue, &evt, 0) == pdTRUE) {
            int64_t now = evt.timestamp_us;
            int64_t dt  = now - last_rain_us;

            if (dt > RAIN_DEBOUNCE_US) {
                last_rain_us = now;
                tips_total++;
                tips_window++;
            }
        }

        vTaskDelayUntil(&last_wake, period);

        float rain_mm_window = tips_window * MM_PER_TIP;
        float rain_mm_total  = tips_total * MM_PER_TIP;
        (void)rain_mm_total;

        if (xSemaphoreTake(g_sample_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            g_sample.rainfall = rain_mm_window;
            xSemaphoreGive(g_sample_mutex);
        }

        tips_window = 0;
    }
}

static void wind_speed_task(void *pvParameters)
{
    (void) pvParameters;

    sensor_event_t evt;
    int64_t last_wind_us = 0;

    uint32_t ticks_window = 0;
    const float window_seconds = 1.0f;
    const TickType_t period    = pdMS_TO_TICKS((int)(window_seconds * 1000));
    TickType_t last_wake       = xTaskGetTickCount();

    for (;;) {
        if (xQueueReceive(wind_speed_queue, &evt, pdMS_TO_TICKS(50))) {
            int64_t now = evt.timestamp_us;
            if (now - last_wind_us > WIND_DEBOUNCE_US) {
                last_wind_us = now;
                ticks_window++;
            }
        }

        TickType_t now_ticks = xTaskGetTickCount();
        if (now_ticks - last_wake >= period) {
            last_wake = now_ticks;

            float cps      = ticks_window / window_seconds;
            float wind_kph = cps * KPH_PER_HZ;

            if (xSemaphoreTake(g_sample_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                g_sample.wind_speed = wind_kph;
                xSemaphoreGive(g_sample_mutex);
            }

            ticks_window = 0;
        }
    }
}

void sht21_task(void *pvParameters)
{
    (void) pvParameters;

    sht21_measurements_t measurements;
    for (;;) {
        esp_err_t err = sht21_read(&measurements);
        if (err == ESP_OK) {
            if (xSemaphoreTake(g_sample_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                g_sample.temperature = measurements.temperature;
                g_sample.humidity    = measurements.humidity;
                xSemaphoreGive(g_sample_mutex);
            }
        } else {
            ESP_LOGE(TAG, "Failed to read from SHT21 sensor");
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void ms5611_task(void *pvParameters)
{
    (void) pvParameters;

    if (!g_ms5611_available) {
        ESP_LOGW(TAG, "MS5611 not available, task exiting");
        vTaskDelete(NULL);
    }

    float temperature = 0.0f;
    int32_t pressure = 0;
    esp_err_t res;

    for (;;) {
        res = ms5611_get_sensor_data(&ms_dev, &pressure, &temperature);
        if (res != ESP_OK) {
            ESP_LOGE(TAG, "MS5611 read failed: %d (%s)",
                     res, esp_err_to_name(res));
        } else {
            float press_hpa  = pressure / 100.0f;
            float press_mmHg = press_hpa * 0.750061683f;

            if (xSemaphoreTake(g_sample_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                g_sample.pressure = press_mmHg;
                g_sample.altitude = calculate_altitude_m(pressure, 1013.25f);
                xSemaphoreGive(g_sample_mutex);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void gp2y_task(void *pvParameters)
{
    (void) pvParameters;

    for (;;) {
        const int N = 20;
        float sum_v = 0.0f;
        for (int i = 0; i < N; i++) {
            float v = gp2y_read_voltage();
            sum_v += v;
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        float v_avg = sum_v / N;
        float dust_ugm3 = gp2y_voltage_to_density_ugm3(v_avg);

        if (xSemaphoreTake(g_sample_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            g_sample.dust = dust_ugm3;
            xSemaphoreGive(g_sample_mutex);
        }

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

// ------------ REST-API task ------------

static void rest_worker_task(void *arg)
{
    (void)arg;

    char jwt_buf[512];
    esp_err_t err = generate_jwt(jwt_buf, sizeof(jwt_buf));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "generate_jwt failed: %s", esp_err_to_name(err));
        vTaskDelete(NULL);
    }

    ESP_LOGI(TAG, "JWT buf: '%s'", jwt_buf);

    header_jwt = strdup(jwt_buf);
    if (!header_jwt) {
        ESP_LOGE(TAG, "Failed to allocate initial JWT");
        vTaskDelete(NULL);
    }

    ESP_LOGI(TAG, "Initial JWT generated");

    CURRENT_STATE = SEND_STATE;
    current_url   = SEND_DATA_URL;

    vTaskDelay(pdMS_TO_TICKS(3000)); // Wait for Wi-Fi

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(POST_PERIOD_MS));

        meteo_sample_t sample_copy;

        if (xSemaphoreTake(g_sample_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            sample_copy = g_sample;
            xSemaphoreGive(g_sample_mutex);
        } else {
            ESP_LOGW(TAG, "Failed to take sample mutex, skipping send");
            continue;
        }

        if (!current_url || !header_jwt) {
            ESP_LOGW(TAG, "No URL or JWT yet, skipping send");
            continue;
        }

        ESP_LOGI(TAG, "Sending segment to %s (state %d)", current_url, CURRENT_STATE);

        char *response = NULL;
        int code = send_segment(current_url, header_jwt, &response, &sample_copy);

        switch (CURRENT_STATE) {
        case SEND_STATE:
            if (code == 200) {
                ESP_LOGI(TAG, "Data sent successfully: %s", response ? response : "(no body)");
            }
            if (code == 405) {
                ESP_LOGW(TAG, "405 from /get_data, switching to GENERATE_TOKEN_STATE");
                CURRENT_STATE = GENERATE_TOKEN_STATE;
                current_url   = GENERATE_TOKEN_URL;
            }
            if (code == 403) {
                ESP_LOGW(TAG, "403 from /get_data: %s", response ? response : "(no body)");
            }
            if (code == 400) {
                ESP_LOGW(TAG, "400 from /get_data: %s", response ? response : "(no body)");
            }
            break;

        case GENERATE_TOKEN_STATE:
            if (code == 200) {
                ESP_LOGI(TAG, "New token received: %s", response ? response : "(no body)");
                if (header_jwt) free(header_jwt);
                if (response) {
                    header_jwt = strdup(response);
                } else {
                    header_jwt = NULL;
                }
                CURRENT_STATE = SEND_STATE;
                current_url   = SEND_DATA_URL;
            }
            if (code == 403) {
                ESP_LOGW(TAG, "403 from /generate_token: %s", response ? response : "(no body)");
                CURRENT_STATE = REQUEST_ACCESS_STATE;
                current_url   = REQUEST_ACCESS_URL;
            }
            if (code == 405) {
                ESP_LOGW(TAG, "405 from /generate_token: %s", response ? response : "(no body)");
            }
            break;

        case REQUEST_ACCESS_STATE:
            if (code == 200) {
                ESP_LOGI(TAG, "Access request response: %s", response ? response : "(no body)");
                CURRENT_STATE = GENERATE_TOKEN_STATE;
                current_url   = GENERATE_TOKEN_URL;
            }
            if (code == 400) {
                ESP_LOGW(TAG, "400 from /request: %s", response ? response : "(no body)");
            }
            if (code == 405) {
                ESP_LOGW(TAG, "405 from /request: %s", response ? response : "(no body)");
            }
            break;
        }

        if (response) {
            free(response);
        }
    }
}

// ------------ app_main ------------

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    wifi_init_sta();

    ESP_ERROR_CHECK(i2cdev_init());
    init_led();
    init_adc();

    g_sample_mutex = xSemaphoreCreateMutex();
    configASSERT(g_sample_mutex != NULL);
    memset(&g_sample, 0, sizeof(g_sample));

    rain_queue       = xQueueCreate(10, sizeof(sensor_event_t));
    wind_speed_queue = xQueueCreate(10, sizeof(sensor_event_t));
    configASSERT(rain_queue != NULL);
    configASSERT(wind_speed_queue != NULL);

    ESP_ERROR_CHECK(sht21_init(I2C_MASTER_PORT, 0x40,
                               100 / portTICK_PERIOD_MS));

    ESP_ERROR_CHECK(ms5611_init_desc(&ms_dev, 0x77,
                                     I2C_MASTER_PORT, I2C_MASTER_SDA_IO,
                                     I2C_MASTER_SCL_IO));

    vTaskDelay(pdMS_TO_TICKS(100));

    esp_err_t ms_err = ms5611_init(&ms_dev, MS5611_OSR_1024);
    if (ms_err != ESP_OK) {
        ESP_LOGE(TAG, "MS5611 init failed: %s - sensor may not be connected", esp_err_to_name(ms_err));
        g_ms5611_available = false;
    } else {
        ESP_LOGI(TAG, "MS5611 initialized successfully");
        g_ms5611_available = true;
    }

    init_isr();

    xTaskCreate(rain_task,           "rain_task",           2048, NULL, 7, NULL);
    xTaskCreate(wind_speed_task,     "wind_speed_task",     2048, NULL, 7, NULL);
    xTaskCreate(wind_direction_task, "wind_direction_task", 2048, NULL, 5, NULL);
    xTaskCreate(sht21_task,          "sht21_task",          4096, NULL, 5, NULL);
    xTaskCreate(ms5611_task,         "ms5611_task",         4096, NULL, 5, NULL);
    xTaskCreate(gp2y_task,           "gp2y_task",           4096, NULL, 5, NULL);
    xTaskCreate(rest_worker_task,    "rest_worker_task",    8192, NULL, 5, NULL);
}
