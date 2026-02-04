#include "ms5611.h"
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

static const char *TAG = "MS5611";

#define MS5611_CMD_RESET              0x1E
#define MS5611_CMD_CONV_D1_OSR4096    0x48
#define MS5611_CMD_CONV_D2_OSR4096    0x58
#define MS5611_CMD_ADC_READ           0x00

static esp_err_t ms5611_write_cmd(ms5611_t *dev, uint8_t cmd)
{
    i2c_cmd_handle_t c = i2c_cmd_link_create();
    i2c_master_start(c);
    i2c_master_write_byte(c, (dev->addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(c, cmd, true);
    i2c_master_stop(c);
    esp_err_t ret = i2c_master_cmd_begin(dev->port, c, dev->timeout);
    i2c_cmd_link_delete(c);
    return ret;
}

// ONLY for ADC (D1/D2)
static esp_err_t ms5611_read_bytes(ms5611_t *dev, uint8_t cmd, uint8_t *data, size_t len)
{
    i2c_cmd_handle_t c = i2c_cmd_link_create();
    i2c_master_start(c);
    i2c_master_write_byte(c, (dev->addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(c, cmd, true);
    i2c_master_start(c);
    i2c_master_write_byte(c, (dev->addr << 1) | I2C_MASTER_READ, true);

    if (len > 1) {
        i2c_master_read(c, data, len - 1, I2C_MASTER_ACK);
    }
    i2c_master_read_byte(c, data + len - 1, I2C_MASTER_NACK);

    i2c_master_stop(c);
    esp_err_t ret = i2c_master_cmd_begin(dev->port, c, dev->timeout);
    i2c_cmd_link_delete(c);
    return ret;
}

// PROM: use the sequence you just proved works
static esp_err_t ms5611_read_prom_word(ms5611_t *dev, uint8_t prom_cmd, uint16_t *out)
{
    uint8_t buf[2];
    i2c_cmd_handle_t c = i2c_cmd_link_create();

    i2c_master_start(c);
    i2c_master_write_byte(c, (dev->addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(c, prom_cmd, true);          // e.g. 0xA2
    i2c_master_start(c);
    i2c_master_write_byte(c, (dev->addr << 1) | I2C_MASTER_READ, true);
    i2c_master_read(c, buf, 2, I2C_MASTER_LAST_NACK);
    i2c_master_stop(c);

    esp_err_t ret = i2c_master_cmd_begin(dev->port, c, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(c);
    ESP_LOGI("MS5611_PROM", "cmd=0x%02X ret=%d buf=%02X %02X", prom_cmd, ret, buf[0], buf[1]);
    if (ret != ESP_OK) return ret;

    *out = ((uint16_t)buf[0] << 8) | buf[1];
    return ESP_OK;
}

static esp_err_t ms5611_read_prom(ms5611_t *dev)
{
    uint16_t coeffs[6];

    for (int i = 0; i < 6; i++) {
        uint8_t cmd = 0xA2 + (uint8_t)(i * 2);   // 0xA2..0xAC (C1..C6)
        esp_err_t ret = ms5611_read_prom_word(dev, cmd, &coeffs[i]);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "PROM read failed at index %d, err=%d", i + 1, ret);
            return ret;
        }
        ESP_LOGI(TAG, "PROM[%d] = 0x%04X", i + 1, coeffs[i]);
    }

    dev->C1 = coeffs[0];
    dev->C2 = coeffs[1];
    dev->C3 = coeffs[2];
    dev->C4 = coeffs[3];
    dev->C5 = coeffs[4];
    dev->C6 = coeffs[5];

    ESP_LOGI(TAG, "PROM C1..C6: %u %u %u %u %u %u",
             dev->C1, dev->C2, dev->C3, dev->C4, dev->C5, dev->C6);

    return ESP_OK;
}

static esp_err_t ms5611_read_adc(ms5611_t *dev, uint8_t conv_cmd, uint32_t *out)
{
    esp_err_t ret = ms5611_write_cmd(dev, conv_cmd);
    if (ret != ESP_OK) return ret;

    // worst case for OSR4096 ~9ms; we wait 10ms [web:71][web:74]
    vTaskDelay(pdMS_TO_TICKS(10));

    uint8_t buf[3];
    ret = ms5611_read_bytes(dev, MS5611_CMD_ADC_READ, buf, 3);
    if (ret != ESP_OK) return ret;

    *out = ((uint32_t)buf[0] << 16) | ((uint32_t)buf[1] << 8) | buf[2];
    return ESP_OK;
}

esp_err_t ms5611_init(ms5611_t *dev, i2c_port_t port, uint8_t addr, TickType_t timeout)
{
    ESP_LOGI("MS5611_INIT", "ENTER ms5611_init dev=%p port=%d addr=0x%02X", dev, port, addr);
    if (!dev) return ESP_ERR_INVALID_ARG;

    dev->port = port;
    dev->addr = addr;
    dev->timeout = pdMS_TO_TICKS(100);

    // reset
    esp_err_t ret = ms5611_write_cmd(dev, MS5611_CMD_RESET);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Reset failed");
        return ret;
    }
    vTaskDelay(pdMS_TO_TICKS(3));

    // read PROM
    ret = ms5611_read_prom(dev);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "PROM read failed");
        return ret;
    }

    return ESP_OK;
}

esp_err_t ms5611_read(ms5611_t *dev, float *temperature, float *pressure)
{
    if (!dev || !temperature || !pressure) return ESP_ERR_INVALID_ARG;

    uint32_t D1 = 0, D2 = 0;
    esp_err_t ret;

    ret = ms5611_read_adc(dev, MS5611_CMD_CONV_D1_OSR4096, &D1);
    if (ret != ESP_OK) return ret;

    ret = ms5611_read_adc(dev, MS5611_CMD_CONV_D2_OSR4096, &D2);
    if (ret != ESP_OK) return ret;

    // Datasheet integer compensation [web:71][web:74]
    int32_t dT   = (int32_t)D2 - ((int32_t)dev->C5 * 256);
    int32_t TEMP = 2000 + (int32_t)(((int64_t)dT * dev->C6) / 8388608);
    int64_t OFF  = ((int64_t)dev->C2 * 65536) + (((int64_t)dev->C4 * dT) / 128);
    int64_t SENS = ((int64_t)dev->C1 * 32768) + (((int64_t)dev->C3 * dT) / 256);
    int32_t P    = (int32_t)(((((int64_t)D1 * SENS) / 2097152) - OFF) / 32768);

    float temp_c    = TEMP / 100.0f;
    float press_hpa = P / 100.0f;                 // 0.01 mbar → hPa [web:71][web:74]
    float press_mmHg = press_hpa * 0.750061683f;  // hPa → mmHg [web:76][web:79]

    ESP_LOGI("MS5611_DRIVER_DEBUG",
             "D1=%u, D2=%u, dT=%ld, TEMP=%ld, OFF=%lld, SENS=%lld, P=%ld, %.2f hPa, %.2f mmHg",
             D1, D2, (long)dT, (long)TEMP, (long long)OFF, (long long)SENS, (long)P,
             press_hpa, press_mmHg);

    *temperature = temp_c;
    *pressure    = press_hpa;   // change to press_mmHg if you want mmHg as return

    return ESP_OK;
}
