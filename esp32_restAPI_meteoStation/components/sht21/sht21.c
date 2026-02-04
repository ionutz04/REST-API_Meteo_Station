#include "sht21.h"
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <i2cdev.h>

static const char *TAG = "SHT21";

static i2c_dev_t s_dev;
static TickType_t s_timeout;

static float sht21_convert_temp(uint16_t raw)
{
    return -46.85f + 175.72f * ((float)raw / 65536.0f);
}

static float sht21_convert_hum(uint16_t raw)
{
    return -6.0f + 125.0f * ((float)raw / 65536.0f);
}

esp_err_t sht21_init(i2c_port_t port, uint8_t addr, TickType_t timeout)
{
    s_timeout = timeout;

    memset(&s_dev, 0, sizeof(s_dev));
    s_dev.port = port;
    s_dev.addr = addr;

    // Match MS5611 bus pins: SDA=21, SCL=22
    s_dev.cfg.sda_io_num = 21;
    s_dev.cfg.scl_io_num = 22;
    s_dev.cfg.sda_pullup_en = GPIO_PULLUP_ENABLE;
    s_dev.cfg.scl_pullup_en = GPIO_PULLUP_ENABLE;
    s_dev.cfg.master.clk_speed = 400000;  // or 100000 if you prefer

    ESP_LOGI(TAG, "SHT21 descriptor set: addr=0x%02X, port=%d, SDA=%d, SCL=%d",
             addr, port, s_dev.cfg.sda_io_num, s_dev.cfg.scl_io_num);
    return ESP_OK;
}

static esp_err_t sht21_write_cmd(uint8_t cmd)
{
    return i2c_dev_write(&s_dev, NULL, 0, &cmd, 1);
}

static esp_err_t sht21_read_raw(uint8_t *buf, size_t len)
{
    return i2c_dev_read(&s_dev, NULL, 0, buf, len);
}

esp_err_t sht21_read(sht21_measurements_t *out)
{
    if (!out) return ESP_ERR_INVALID_ARG;

    uint8_t buf[3];
    uint16_t raw;
    esp_err_t err;

    // Temperature
    err = sht21_write_cmd(0xF3);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error sending temperature cmd: %d", err);
        return err;
    }
    vTaskDelay(s_timeout);

    err = sht21_read_raw(buf, 3);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error reading temperature data: %d", err);
        return err;
    }
    raw = ((uint16_t)buf[0] << 8) | buf[1];
    float temperature = sht21_convert_temp(raw);

    // Humidity
    err = sht21_write_cmd(0xF5);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error sending humidity cmd: %d", err);
        return err;
    }
    vTaskDelay(s_timeout);

    err = sht21_read_raw(buf, 3);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error reading humidity data: %d", err);
        return err;
    }
    raw = ((uint16_t)buf[0] << 8) | buf[1];
    float humidity = sht21_convert_hum(raw);

    out->temperature = temperature;
    out->humidity = humidity;

    return ESP_OK;
}
