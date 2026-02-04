#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"

#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_netif.h"

#include "esp_http_client.h"
#include "mbedtls/md.h"
#include "mbedtls/base64.h"
#include "esp_mac.h"

static const char *TAG = "rest_fsm";

// ------------ CONFIG ------------

static const char *WIFI_SSID   = "METEO_AP";
static const char *WIFI_PASS   = "meteo_jetson_nano_passwd";

static const char *JWT_SECRET  = "V3fryS3cr3tK3y,7h47Y0uC4n7F1nd:)!";

// Use plain HTTP on the AP; adjust IP/port/path to your server
static const char *SEND_DATA_URL      = "https://192.168.50.1:5500/get_data";
static const char *GENERATE_TOKEN_URL = "https://192.168.50.1:5500/generate_token";
static const char *REQUEST_ACCESS_URL = "https://192.168.50.1:5500/request";

#define POST_PERIOD_MS 5000

// ------------ STATE MACHINE ------------

typedef enum {
    SEND_STATE = 0,
    GENERATE_TOKEN_STATE,
    REQUEST_ACCESS_STATE
} current_state_t;

static current_state_t CURRENT_STATE = SEND_STATE;
static const char *current_url = NULL;


static char *header_jwt = NULL;



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

    // copy and apply URL-safe transform
    size_t oi = 0;
    for (size_t i = 0; i < b64_len && oi + 1 < out_size; ++i) {
        char c = (char)tmp[i];
        if (c == '+') c = '-';
        else if (c == '/') c = '_';
        else if (c == '=') { break; }  // strip padding

        out[oi++] = c;
    }
    out[oi] = '\0';
    free(tmp);
    return ESP_OK;
}

// base64url from string
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
                        char **response_out)
{
    if (response_out) {
        *response_out = NULL;
    }

    char jwt_encoded[1024];
    url_encode(jwt, jwt_encoded, sizeof(jwt_encoded));

    char url[512];
    snprintf(url, sizeof(url), "%s?jwt=%s", base_url, jwt_encoded);

    // Dummy data – plug your real sensors here
    float temperature = 23.5f;
    float humidity    = 45.2f;
    float wind_speed  = 3.7f;
    float rainfall    = 0.0f;
    float wind_deg    = 180.0f;
    float wind_v      = 1.23f;

    char json[512];
    snprintf(json, sizeof(json),
            "{"
            "\"temperature\":%.2f,"
            "\"humidity\":%.2f,"
            "\"wind_speed\":%.2f,"
            "\"rainfall\":%.1f,"
            "\"wind_direction_degrees\":%.2f,"
            "\"wind_direction_voltage\":%.2f,"
            "\"ssid\":\"%s\""
            "}",
            temperature,
            humidity,
            wind_speed,
            rainfall,
            wind_deg,
            wind_v,
            WIFI_SSID);


    esp_http_client_config_t cfg = {
        .url = url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 10000,
        .transport_type = HTTP_TRANSPORT_OVER_TCP, // plain HTTP
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
                    *response_out = buf; // caller must free
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


static void rest_worker_task(void *arg)
{
    // Generate initial JWT with real chip_id (Arduino: in setup())
    char jwt_buf[512];
    
    ESP_ERROR_CHECK(generate_jwt(jwt_buf, sizeof(jwt_buf)));
    ESP_LOGI(TAG, "JWT buf: '%s'", jwt_buf);   // DEBUG
    header_jwt = strdup(jwt_buf);
    if (!header_jwt) {
        ESP_LOGE(TAG, "Failed to allocate initial JWT");
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "Initial JWT generated");

    CURRENT_STATE = SEND_STATE;
    current_url   = SEND_DATA_URL;

    // Wait for Wi-Fi
    vTaskDelay(pdMS_TO_TICKS(3000));

    while (1) {
        ESP_LOGI(TAG, "Sending segment to %s (state %d)", current_url, CURRENT_STATE);

        char *response = NULL;
        int code = send_segment(current_url, header_jwt, &response);

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
                // free old header_jwt and replace with new token
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

        vTaskDelay(pdMS_TO_TICKS(POST_PERIOD_MS));
    }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    wifi_init_sta();

    xTaskCreate(rest_worker_task, "rest_worker_task", 8192, NULL, 5, NULL);
}
