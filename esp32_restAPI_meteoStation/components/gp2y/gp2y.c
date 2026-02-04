#include "gp2y.h"

static const char *TAG = "GP2Y";

static adc_cali_handle_t adc_cali_handle = NULL;
static bool cali_enabled = false;

esp_err_t gp2y_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << GP2Y_LED_PIN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    esp_err_t ret = gpio_config(&io_conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure LED GPIO");
        return ret;
    }
    gpio_set_level(GP2Y_LED_PIN, 0);

    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(GP2Y_ADC_CHANNEL, GP2Y_ADC_ATTEN);

#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    adc_cali_curve_fitting_config_t cali_config = {
        .unit_id = ADC_UNIT_1,
        .atten = GP2Y_ADC_ATTEN,
        .bitwidth = ADC_BITWIDTH_12,
    };
    ret = adc_cali_create_scheme_curve_fitting(&cali_config, &adc_cali_handle);
    if (ret == ESP_OK) {
        cali_enabled = true;
        ESP_LOGI(TAG, "ADC calibration: Curve Fitting");
    }
#elif ADC_CALI_SCHEME_LINE_FITTING_SUPPORTED
    adc_cali_line_fitting_config_t cali_config = {
        .unit_id = ADC_UNIT_1,
        .atten = GP2Y_ADC_ATTEN,
        .bitwidth = ADC_BITWIDTH_12,
    };
    ret = adc_cali_create_scheme_line_fitting(&cali_config, &adc_cali_handle);
    if (ret == ESP_OK) {
        cali_enabled = true;
        ESP_LOGI(TAG, "ADC calibration: Line Fitting");
    }
#endif

    if (!cali_enabled) {
        ESP_LOGW(TAG, "ADC calibration not available");
    }

    ESP_LOGI(TAG, "GP2Y sensor initialized");
    return ESP_OK;
}

void gp2y_deinit(void)
{
    if (cali_enabled && adc_cali_handle) {
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
        adc_cali_delete_scheme_curve_fitting(adc_cali_handle);
#elif ADC_CALI_SCHEME_LINE_FITTING_SUPPORTED
        adc_cali_delete_scheme_line_fitting(adc_cali_handle);
#endif
        adc_cali_handle = NULL;
        cali_enabled = false;
    }
    ESP_LOGI(TAG, "GP2Y sensor deinitialized");
}

float gp2y_read_voltage(void)
{
    uint32_t adc_reading = 0;

    gpio_set_level(GP2Y_LED_PIN, 1);
    ets_delay_us(280);

    for (int i = 0; i < GP2Y_NO_OF_SAMPLES; i++) {
        adc_reading += adc1_get_raw(GP2Y_ADC_CHANNEL);
    }
    adc_reading /= GP2Y_NO_OF_SAMPLES;

    gpio_set_level(GP2Y_LED_PIN, 0);

    float v = 0.0f;
    if (cali_enabled && adc_cali_handle) {
        int voltage_mv = 0;
        adc_cali_raw_to_voltage(adc_cali_handle, (int)adc_reading, &voltage_mv);
        v = voltage_mv / 1000.0f;
    } else {
        v = (adc_reading / 4095.0f) * 3.3f;
    }

    ESP_LOGI(TAG, "raw=%" PRIu32 ", V=%.3f V", adc_reading, v);
    return v;
}

float gp2y_voltage_to_density_ugm3(float vo)
{
    float density_mg_m3 = 0.170f * vo - 0.1f;
    if (density_mg_m3 < 0.0f) density_mg_m3 = 0.0f;
    return density_mg_m3 * 1000.0f;
}

float gp2y_read_dust_density(void)
{
    float voltage = gp2y_read_voltage();
    return gp2y_voltage_to_density_ugm3(voltage);
}
