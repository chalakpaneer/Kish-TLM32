idf_component_register(
    SRCS "src/oled.c"
    INCLUDE_DIRS "include"
    REQUIRES driver esp_driver_i2c
)
