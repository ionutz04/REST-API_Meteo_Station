#ifndef GP2Y_H
#define GP2Y_H

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "driver/gpio.h"
#include "esp_system.h"
#include "esp_log.h"
#include "rom/ets_sys.h"

#define GP2Y_LED_PIN        4
#define GP2Y_ADC_CHANNEL    ADC1_CHANNEL_6
#define GP2Y_ADC_ATTEN      ADC_ATTEN_DB_11
#define GP2Y_NO_OF_SAMPLES  64

esp_err_t gp2y_init(void);
void gp2y_deinit(void);
float gp2y_read_voltage(void);
float gp2y_voltage_to_density_ugm3(float vo);
float gp2y_read_dust_density(void);

#endif