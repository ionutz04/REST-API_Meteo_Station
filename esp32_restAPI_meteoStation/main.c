#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "driver/adc.h"
#include "esp_adc_cal.h"
#include "driver/gpio.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "rom/ets_sys.h" 
#include <driver/i2c.h>
#include <freertos/queue.h>


#include "sht21.h"
#include <i2cdev.h>
#include <ms5611.h>
#include <inttypes.h> 


#define I2C_MASTER_SDA_IO   21
#define I2C_MASTER_SCL_IO   22
#define I2C_MASTER_PORT     I2C_NUM_0
#define I2C_MASTER_FREQ_HZ  100000

static const char *TAG = "main";

void sht21_task(void *pvParameters);
void ms5611_task(void *pvParameters);
void gp2y_task(void *pvParameters);
static ms5611_t ms_dev;

const gpio_num_t rain_pin = GPIO_NUM_33;
const gpio_num_t wind_speed_pin = GPIO_NUM_32;

#define LED_PIN        4                  // LED pin to GP2Y LED input
#define ADC_CHANNEL    ADC1_CHANNEL_6     // GPIO34 (ADC1_CH6)
#define DEFAULT_VREF   1100               // mV
#define NO_OF_SAMPLES  64                 // ADC averaging
#define VCC_SENSOR     5.0f               // sensor supply for formula (if Vo is scaled to 0–VCC_SENSOR)

// If your module already scales Vo to 0–3.3 V, set this accordingly
#define ADC_FULL_SCALE 3.3f               // volts corresponding to ADC full scale at this attenuation

static esp_adc_cal_characteristics_t *adc_chars = NULL;

volatile unsigned long rotatii = 0;
volatile unsigned long tipsBucket = 0;
volatile unsigned long lastTipTime = 0;
const unsigned long DEBOUNCE_MS = 100;  // Debounce for rain gauge switch
const float MM_PER_TIP = 0.2794;  // mm of rain per bucket tip
unsigned long ultimaMasuraRain = 0;
unsigned long ultimaMasuraWind = 0;
// ADC init
static void init_adc(void)
{
    ESP_ERROR_CHECK(adc1_config_width(ADC_WIDTH_BIT_12));
    // For a GP2Y module that outputs up to ~3.3 V use 11 dB; if you are sure Vo < 1.1 V, 0 dB is ok.
    ESP_ERROR_CHECK(adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN_DB_11));

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
        .pin_bit_mask = (1ULL << LED_PIN), 
        .mode = GPIO_MODE_OUTPUT,      
        .pull_up_en = GPIO_PULLUP_DISABLE,  
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE     
    };

    gpio_config(&io_conf);


    gpio_set_level(LED_PIN, 0);
}



static float gp2y_read_voltage(void)
{
    uint32_t adc_reading = 0;

    // Turn LED ON
    gpio_set_level(LED_PIN, 1);
    ets_delay_us(280);  


    for (int i = 0; i < NO_OF_SAMPLES; i++) {
        adc_reading += adc1_get_raw(ADC_CHANNEL);
    }
    adc_reading /= NO_OF_SAMPLES;


    gpio_set_level(LED_PIN, 0);

    uint32_t mv = esp_adc_cal_raw_to_voltage(adc_reading, adc_chars);
    float v = mv / 1000.0f; 
    return v;
}



static float gp2y_voltage_to_density_ugm3(float vo)
{
    // vo is sensor output in volts, assumed proportional to dust and referenced to VCC_SENSOR
    float density_mg_m3 = 0.170f * vo - 0.1f;   // mg/m3
    if (density_mg_m3 < 0.0f) density_mg_m3 = 0.0f;
    return density_mg_m3 * 1000.0f;             // ug/m3
}

// Optional PROM test using raw IDF I2C
static void ms5611_prom_test(void)
{
    uint8_t addr = 0x77;
    uint8_t cmd  = 0xA2;
    uint8_t buf[2];

    i2c_cmd_handle_t c = i2c_cmd_link_create();
    i2c_master_start(c);
    i2c_master_write_byte(c, (addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(c, cmd, true);
    i2c_master_start(c);
    i2c_master_write_byte(c, (addr << 1) | I2C_MASTER_READ, true);
    i2c_master_read(c, buf, 2, I2C_MASTER_LAST_NACK);
    i2c_master_stop(c);

    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_PORT, c,
                                         pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(c);

}


static QueueHandle_t gpio_event_queue = NULL;

typedef enum{
    RAIN_EVENT,
    WIND_SPEED_EVENT
}sensor_event_type_t;

typedef struct{
    sensor_event_type_t type;
    int timestamp_us;
}sensor_event_t;

static void IRAM_ATTR rain_isr_handler(void* arg){
    sensor_event_t evt = {
        .type = RAIN_EVENT,
        .timestamp_us = (int)esp_timer_get_time()
    };
    BaseType_t hpw = pdFALSE;
    xQueueSendFromISR(gpio_event_queue, &evt, &hpw);
    if(hpw){
        portYIELD_FROM_ISR();
    }
}

static void IRAM_ATTR wind_speed_isr_handler(void* arg){
    sensor_event_t evt = {
        .type = WIND_SPEED_EVENT,
        .timestamp_us = (int)esp_timer_get_time()
    };
    BaseType_t hpw = pdFALSE;
    xQueueSendFromISR(gpio_event_queue, &evt, &hpw);
    if(hpw){
        portYIELD_FROM_ISR();
    }
}
void init_isr(){
    gpio_config_t io_conf = {
        .intr_type = GPIO_INTR_NEGEDGE,
        .mode = GPIO_MODE_INPUT,
        .pin_bit_mask = (1ULL << rain_pin) | (1ULL << wind_speed_pin),
        .pull_up_en = GPIO_PULLUP_ENABLE,
    };
    gpio_config(&io_conf);

    gpio_event_queue = xQueueCreate(10, sizeof(sensor_event_t));

    gpio_install_isr_service(0);
    gpio_isr_handler_add(rain_pin, rain_isr_handler, (void*) rain_pin);
    gpio_isr_handler_add(wind_speed_pin, wind_speed_isr_handler, (void*) wind_speed_pin);
}

void sensor_task(void* pvParameters);
void rain_task(void* pvParameters);
void wind_speed_task(void* pvParameters);


void app_main(void)
{
    init_isr();

    ESP_ERROR_CHECK(i2cdev_init());
    init_led();
    init_adc();
    // MS5611
    ESP_ERROR_CHECK(ms5611_init_desc(&ms_dev, 0x77,
                                     I2C_MASTER_PORT, I2C_MASTER_SDA_IO,
                                     I2C_MASTER_SCL_IO));
    ESP_ERROR_CHECK(ms5611_init(&ms_dev, MS5611_OSR_1024));

    // SHT21 (uses same port, address 0x40)
    ESP_ERROR_CHECK(sht21_init(I2C_MASTER_PORT, 0x40,
                               100 / portTICK_PERIOD_MS));


    xTaskCreate(sht21_task, "sht21_task", 4096, NULL, 5, NULL);
    xTaskCreate(ms5611_task, "ms5611_task", 4096, NULL, 5, NULL);
    xTaskCreate(gp2y_task, "gp2y_task", 4096, NULL, 5, NULL);
    xTaskCreate(rain_task, "rain_task", 2048, NULL, 6, NULL);
    xTaskCreate(sensor_task, "sensor_task", 4096, NULL, 7, NULL);
    xTaskCreate(wind_speed_task, "wind_speed_task", 2048, NULL, 6, NULL);
}

typedef struct {
    float voltage;
    const char* cardinal;
    float degrees;
}WindDirection;

// Sorted by voltage for easier matching
const WindDirection directions[] = {
    {0.00, "NE",  45.0},   // NE alternate (two switches)
    {0.14, "NW", 315.0},   // NW alternate (two switches)
    {0.18, "N",    0.0},   // North
    {0.30, "E",   90.0},   // East alternate (two switches)
    {0.50, "NE",  45.0},   // NE primary
    {1.00, "E",   90.0},   // East primary
    {1.50, "NW", 315.0},   // NW primary
    {2.00, "SE", 135.0},   // SE
    {2.50, "W",  270.0},   // West
    {2.60, "SW", 225.0},   // SW alternate
    {2.70, "SW", 225.0},   // SW primary
    {2.80, "S",  180.0},   // South
};
const int numDirections = sizeof(directions) / sizeof(directions[0]);

const char* getWindDirection(float voltage, float* degrees) {
    float minDiff = 999.0;
    int bestMatch = 0;
    
    for (int i = 0; i < numDirections; i++) {
        float diff = abs(voltage - directions[i].voltage);
        if (diff < minDiff) {
            minDiff = diff;
            bestMatch = i;
        }
    }
    
    *degrees = directions[bestMatch].degrees;
    return directions[bestMatch].cardinal;
}
static float debit = 0.0f;

void sensor_task(void* pvParameters)
{
    sensor_event_t evt;
    while (1) {
        if (xQueueReceive(gpio_event_queue, &evt, portMAX_DELAY)) {
            switch (evt.type) {
                case RAIN_EVENT:
                    if(evt.timestamp_us - ultimaMasuraRain > DEBOUNCE_MS * 1000) {
                        tipsBucket++;
                        ultimaMasuraRain = evt.timestamp_us;
                    }
                    break;
                case WIND_SPEED_EVENT:
                    if(evt.timestamp_us - ultimaMasuraWind > DEBOUNCE_MS * 1000) {
                        rotatii++;
                        ultimaMasuraWind = evt.timestamp_us;
                    }
                    break;
                default:
                    break;
            }
        }
    }
}

void rain_task(void* pvParameters)
{
    (void) pvParameters;
    while(1){
        vTaskDelay(pdMS_TO_TICKS(5000)); // 5 seconds
        float rain_mm = tipsBucket * MM_PER_TIP;
        ESP_LOGI(TAG, "Rain: %.2f ml/m^3", rain_mm);
        tipsBucket = 0;
    }
}
void wind_speed_task(void* pvParameters)
{
    (void) pvParameters;
    while(1){
        vTaskDelay(pdMS_TO_TICKS(5000)); // 5 seconds
        float windSpeed = (rotatii * 2.4f); // km/h
        ESP_LOGI(TAG, "Wind Speed: %.2f km/h", windSpeed);
        rotatii = 0;
    }
}
void sht21_task(void *pvParameters)
{
    (void) pvParameters;

    sht21_measurements_t measurements;
    while (1) {
        esp_err_t err = sht21_read(&measurements);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "SHT21 - Temperature: %.2f C, Humidity: %.2f %%",
                     measurements.temperature, measurements.humidity);
        } else {
            ESP_LOGE(TAG, "Failed to read from SHT21 sensor");
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void ms5611_task(void *pvParameters)
{
    (void) pvParameters;

    float temperature = 0.0f;
    int32_t pressure = 0;
    esp_err_t res;

    while (1) {
        res = ms5611_get_sensor_data(&ms_dev, &pressure, &temperature);
        if (res != ESP_OK) {
            ESP_LOGE(TAG, "MS5611 read failed: %d (%s)",
                     res, esp_err_to_name(res));
            continue;
        }

        float press_hpa  = pressure / 100.0f;
        float press_mmHg = press_hpa * 0.750061683f;

        ESP_LOGI(TAG,
                 "MS5611 - Pressure: %" PRIi32 " Pa (%.2f hPa, %.2f mmHg), Temperature: %.2f C",
                 pressure, press_hpa, press_mmHg, temperature);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void gp2y_task(void *pvParameters)
{
    while (1) {
        // Optionally average multiple pulses
        const int N = 20;
        float sum_v = 0.0f;
        for (int i = 0; i < N; i++) {
            float v = gp2y_read_voltage();
            sum_v += v;
            vTaskDelay(pdMS_TO_TICKS(10)); // ~10 ms between pulses [web:42][web:5]
        }
        float v_avg = sum_v / N;
        float dust_ugm3 = gp2y_voltage_to_density_ugm3(v_avg);

        ESP_LOGI(TAG, "V_avg=%.3f V, Dust=%.3f ug/m3", v_avg, dust_ugm3);

        vTaskDelay(pdMS_TO_TICKS(1000));  // log every second
    }
}
