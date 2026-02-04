idf_component_register(SRCS "sht21.c"
                      INCLUDE_DIRS "."
                      REQUIRES driver i2cdev)