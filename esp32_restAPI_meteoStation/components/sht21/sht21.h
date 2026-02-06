#ifndef SHT21_H
#define SHT21_H

#include <driver/i2c.h>
#include <esp_err.h>
#include <esp_log.h>
#include <i2cdev.h>

typedef struct{
    float temperature;
    float humidity;
}sht21_measurements_t;

esp_err_t sht21_init(i2c_port_t port, uint8_t addr, TickType_t timeout);

esp_err_t sht21_read(sht21_measurements_t *out);

#endif