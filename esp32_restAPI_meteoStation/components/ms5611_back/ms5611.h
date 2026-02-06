#pragma ones

#include <driver/i2c.h>
#include <esp_err.h>
#include <esp_log.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"{
#endif

typedef struct{
    i2c_port_t port;
    uint8_t addr;
    TickType_t timeout;

    uint16_t C1;
    uint16_t C2;
    uint16_t C3;
    uint16_t C4;
    uint16_t C5;
    uint16_t C6;
}ms5611_t;

esp_err_t ms5611_init(ms5611_t* dev, i2c_port_t port, uint8_t addr, TickType_t timeout);

esp_err_t ms5611_read(ms5611_t* dev, float* temperature, float* pressure);

#ifdef __cplusplus
}
#endif