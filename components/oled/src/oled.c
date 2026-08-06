#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "oled.h"

static const char *TAG = "OLED";

/* ---- Wiring / bus config ---- */
#define OLED_SDA_GPIO   25
#define OLED_SCL_GPIO   26
#define OLED_I2C_PORT   I2C_NUM_0
#define OLED_I2C_ADDR   0x3C
#define OLED_I2C_HZ     400000

#define OLED_WIDTH      128
#define OLED_HEIGHT     64
#define OLED_PAGES      (OLED_HEIGHT / 8)   /* 8 */

static i2c_master_bus_handle_t s_bus = NULL;
static i2c_master_dev_handle_t s_dev = NULL;
static bool s_ready = false;

/* One byte per column per page: bit n = pixel at (col, page*8 + n). */
static uint8_t s_fb[OLED_PAGES][OLED_WIDTH];

/* ---- Minimal 5x7 font ---- see oled.h for exactly which characters
 * are covered and why. Each glyph is 5 bytes, one per column, bit 0 =
 * top row. Index 0 = space (ASCII 32). */
typedef struct { char ch; uint8_t col[5]; } FontGlyph;

static const FontGlyph FONT_5X7[] = {
    {' ', {0x00,0x00,0x00,0x00,0x00}},
    {'!', {0x00,0x00,0x5F,0x00,0x00}},
    {'\'',{0x00,0x05,0x03,0x00,0x00}},
    {'%', {0x23,0x13,0x08,0x64,0x62}},
    {'-', {0x08,0x08,0x08,0x08,0x08}},
    {'.', {0x00,0x60,0x60,0x00,0x00}},
    {'/', {0x20,0x10,0x08,0x04,0x02}},
    {':', {0x00,0x36,0x36,0x00,0x00}},
    {'?', {0x02,0x01,0x59,0x09,0x06}},
    {'0', {0x3E,0x51,0x49,0x45,0x3E}},
    {'1', {0x00,0x42,0x7F,0x40,0x00}},
    {'2', {0x42,0x61,0x51,0x49,0x46}},
    {'3', {0x21,0x41,0x45,0x4B,0x31}},
    {'4', {0x18,0x14,0x12,0x7F,0x10}},
    {'5', {0x27,0x45,0x45,0x45,0x39}},
    {'6', {0x3C,0x4A,0x49,0x49,0x30}},
    {'7', {0x01,0x71,0x09,0x05,0x03}},
    {'8', {0x36,0x49,0x49,0x49,0x36}},
    {'9', {0x06,0x49,0x49,0x29,0x1E}},
    {'A', {0x7E,0x11,0x11,0x11,0x7E}},
    {'B', {0x7F,0x49,0x49,0x49,0x36}},
    {'C', {0x3E,0x41,0x41,0x41,0x22}},
    {'D', {0x7F,0x41,0x41,0x22,0x1C}},
    {'E', {0x7F,0x49,0x49,0x49,0x41}},
    {'F', {0x7F,0x09,0x09,0x09,0x01}},
    {'G', {0x3E,0x41,0x49,0x49,0x7A}},
    {'H', {0x7F,0x08,0x08,0x08,0x7F}},
    {'I', {0x00,0x41,0x7F,0x41,0x00}},
    {'J', {0x20,0x40,0x41,0x3F,0x01}},
    {'K', {0x7F,0x08,0x14,0x22,0x41}},
    {'L', {0x7F,0x40,0x40,0x40,0x40}},
    {'M', {0x7F,0x02,0x0C,0x02,0x7F}},
    {'N', {0x7F,0x04,0x08,0x10,0x7F}},
    {'O', {0x3E,0x41,0x41,0x41,0x3E}},
    {'P', {0x7F,0x09,0x09,0x09,0x06}},
    {'Q', {0x3E,0x41,0x51,0x21,0x5E}},
    {'R', {0x7F,0x09,0x19,0x29,0x46}},
    {'S', {0x46,0x49,0x49,0x49,0x31}},
    {'T', {0x01,0x01,0x7F,0x01,0x01}},
    {'U', {0x3F,0x40,0x40,0x40,0x3F}},
    {'V', {0x1F,0x20,0x40,0x20,0x1F}},
    {'W', {0x3F,0x40,0x38,0x40,0x3F}},
    {'X', {0x63,0x14,0x08,0x14,0x63}},
    {'Y', {0x07,0x08,0x70,0x08,0x07}},
    {'Z', {0x61,0x51,0x49,0x45,0x43}},
};
#define FONT_N (sizeof(FONT_5X7) / sizeof(FONT_5X7[0]))

static const uint8_t *glyph_for(char c)
{
    for (size_t i = 0; i < FONT_N; i++) {
        if (FONT_5X7[i].ch == c) return FONT_5X7[i].col;
    }
    return NULL; /* unsupported char -> caller draws blank */
}

/* ---- Low-level I2C ---- */
static bool oled_cmd(uint8_t cmd)
{
    uint8_t buf[2] = {0x00, cmd}; /* 0x00 = "next byte is a command" */
    return i2c_master_transmit(s_dev, buf, sizeof(buf), 100) == ESP_OK;
}

static bool oled_cmd2(uint8_t cmd, uint8_t arg)
{
    return oled_cmd(cmd) && oled_cmd(arg);
}

static void oled_flush(void)
{
    if (!s_ready) return;

    oled_cmd(0x21); oled_cmd(0);   oled_cmd(OLED_WIDTH - 1);   /* column range */
    oled_cmd(0x22); oled_cmd(0);   oled_cmd(OLED_PAGES - 1);   /* page range   */

    /* Send framebuffer in chunks with a 0x40 ("data") control byte
     * prefix on each I2C transaction. */
    uint8_t chunk[33];
    chunk[0] = 0x40;
    for (int p = 0; p < OLED_PAGES; p++) {
        for (int off = 0; off < OLED_WIDTH; off += 32) {
            memcpy(chunk + 1, &s_fb[p][off], 32);
            i2c_master_transmit(s_dev, chunk, 33, 100);
        }
    }
}

/* ---- Drawing primitives ---- */
void oled_clear(void)
{
    memset(s_fb, 0, sizeof(s_fb));
}

/* Draws text at (page, col_start), uppercased, clipped to the display
 * width. Unsupported characters render as a blank cell rather than
 * garbage. Returns nothing — this is a status display, not a general
 * text widget. */
static void draw_text(int page, int col_start, const char *text)
{
    if (page < 0 || page >= OLED_PAGES) return;
    int col = col_start;
    for (const char *p = text; *p != '\0' && col < OLED_WIDTH - 5; p++) {
        char c = (char)toupper((unsigned char)*p);
        const uint8_t *glyph = glyph_for(c);
        for (int i = 0; i < 5; i++) {
            s_fb[page][col + i] = glyph ? glyph[i] : 0x00;
        }
        col += 6; /* 5 px glyph + 1 px spacing */
    }
}

/* Solid horizontal progress bar spanning the full width at the given
 * page, filled left-to-right proportional to frac (0.0-1.0). */
static void draw_progress_bar(int page, float frac)
{
    if (page < 0 || page >= OLED_PAGES) return;
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 1.0f) frac = 1.0f;
    int filled_cols = (int)(frac * OLED_WIDTH);

    for (int col = 0; col < OLED_WIDTH; col++) {
        if (col == 0 || col == OLED_WIDTH - 1) {
            s_fb[page][col] = 0xFF; /* end caps, full-height border */
        } else if (col < filled_cols) {
            s_fb[page][col] = 0x7E; /* filled: solid middle band */
        } else {
            s_fb[page][col] = 0x42; /* empty: top/bottom border only */
        }
    }
}

/* ---- High-level status screens (this is the only API main.c uses) ---- */

bool oled_init(void)
{
    i2c_master_bus_config_t bus_config = {
        .i2c_port = OLED_I2C_PORT,
        .sda_io_num = OLED_SDA_GPIO,
        .scl_io_num = OLED_SCL_GPIO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    if (i2c_new_master_bus(&bus_config, &s_bus) != ESP_OK) {
        ESP_LOGW(TAG, "i2c_new_master_bus failed");
        return false;
    }

    i2c_device_config_t dev_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = OLED_I2C_ADDR,
        .scl_speed_hz = OLED_I2C_HZ,
    };
    if (i2c_master_bus_add_device(s_bus, &dev_config, &s_dev) != ESP_OK) {
        ESP_LOGW(TAG, "i2c_master_bus_add_device failed");
        return false;
    }

    s_ready = true; /* needed before oled_cmd() will do anything */

    static const uint8_t init_seq[] = {
        0xAE,             /* display off                     */
        0xD5, 0x80,       /* clock divide                    */
        0xA8, 0x3F,       /* multiplex ratio = 64             */
        0xD3, 0x00,       /* display offset = 0               */
        0x40,             /* start line = 0                   */
        0x8D, 0x14,       /* charge pump on                   */
        0x20, 0x00,       /* horizontal addressing mode       */
        0xA1,             /* segment remap                    */
        0xC8,             /* COM scan direction, reversed     */
        0xDA, 0x12,       /* COM pins config                  */
        0x81, 0xCF,       /* contrast                         */
        0xD9, 0xF1,       /* pre-charge period                */
        0xDB, 0x40,       /* VCOMH deselect level              */
        0xA4,             /* resume RAM content display        */
        0xA6,             /* normal (not inverted) display     */
        0xAF,             /* display on                        */
    };
    bool ok = true;
    for (size_t i = 0; i < sizeof(init_seq); i++) {
        ok = oled_cmd(init_seq[i]) && ok;
    }

    if (!ok) {
        ESP_LOGW(TAG, "SSD1306 did not ACK init sequence -- check wiring/address");
        s_ready = false;
        return false;
    }

    oled_clear();
    oled_flush();
    ESP_LOGI(TAG, "OLED ready");
    return true;
}

void oled_boot(void)
{
    if (!s_ready) return;
    oled_clear();
    draw_text(1, 4, "KISH-TLM32");
    draw_text(3, 4, "TINY LLM");
    draw_text(5, 4, "BOOTING...");
    oled_flush();
    vTaskDelay(pdMS_TO_TICKS(2000));
}

void oled_waiting(void)
{
    if (!s_ready) return;
    oled_clear();
    draw_text(1, 4, "KISH-TLM32");
    draw_text(3, 4, "WAITING...");
    draw_text(5, 4, "UART READY");
    oled_flush();
}

void oled_prompt(const char *text)
{
    if (!s_ready) return;
    oled_clear();
    draw_text(0, 0, "PROMPT");
    draw_text(2, 0, text);
    /* if it's longer than one line, wrap onto a second line -- crude
     * but functional for short prompts */
    if (strlen(text) > 21) {
        draw_text(4, 0, text + 21);
    }
    oled_flush();
}

void oled_thinking(int token, int total)
{
    if (!s_ready) return;
    oled_clear();
    draw_text(0, 0, "THINKING...");

    char counter[24];
    snprintf(counter, sizeof(counter), "TOKEN %d / %d", token, total);
    draw_text(2, 0, counter);

    float frac = (total > 0) ? ((float)token / (float)total) : 0.0f;
    draw_progress_bar(5, frac);

    oled_flush();
}

void oled_done(float ms_per_token, int tokens_generated)
{
    if (!s_ready) return;
    oled_clear();
    draw_text(0, 0, "DONE");

    char line1[24], line2[24];
    snprintf(line1, sizeof(line1), "%.0f MS/TOKEN", ms_per_token);
    snprintf(line2, sizeof(line2), "%d TOKENS", tokens_generated);
    draw_text(2, 0, line1);
    draw_text(4, 0, line2);

    oled_flush();
}

void oled_response_preview(const char *text)
{
    if (!s_ready) return;
    /* Draws under whatever oled_done() left on screen -- keep it short,
     * this is a preview, the full response always goes to serial. */
    char preview[22];
    size_t n = strlen(text);
    if (n > 21) n = 21;
    memcpy(preview, text, n);
    preview[n] = '\0';
    draw_text(6, 0, preview);
    oled_flush();
}
