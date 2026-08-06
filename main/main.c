#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_random.h"
#include "tlm_math.h"
#include "tlm_config.h"
#include "tlm_model.h"
#include "tlm_weights_data.h"
#include "oled.h"

static const char *TAG = "KISH-TLM32";

#define MIN_GEN_TOKENS     20
#define MAX_GEN_TOKENS    200    /* automatic upper limit */
#define YIELD_EVERY_N     4    /* feed the watchdog + refresh OLED this often */
#define SAMPLE_TEMPERATURE 0.0f /* 0.0f = deterministic argmax; >0 = sampled */
#define SAMPLE_TOP_K       0    /* 0 = no restriction */
#define LINE_BUF_SIZE    128

/* Bug fix #1 (watchdog): ESP-IDF's IDLE0 task feeds the Task Watchdog Timer,
 * but it only runs when nothing else is ready to run. A tight loop that's
 * all tlm_forward() calls back-to-back starves IDLE0 and the watchdog fires.
 * A cheap vTaskDelay(1) every few tokens is enough to let IDLE0 run.
 */

static void init_console_uart(void)
{
    setvbuf(stdin, NULL, _IONBF, 0);

    uart_vfs_dev_port_set_rx_line_endings(UART_NUM_0, ESP_LINE_ENDINGS_CR);
    uart_vfs_dev_port_set_tx_line_endings(UART_NUM_0, ESP_LINE_ENDINGS_CRLF);

    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_0, &uart_config));
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 256, 0, 0, NULL, 0));
    uart_vfs_dev_use_driver(UART_NUM_0);
}

/* Bug fix #2 (nothing appears until Enter): the previous version read
 * input with fgets(), which blocks silently until a full line arrives.
 * That's fine for GETTING the line, but it means nothing you type is
 * echoed back until you hit Enter -- because nothing in that path ever
 * writes your keystrokes back out over UART. A raw serial line doesn't
 * echo on its own; something has to explicitly send each typed
 * character back out as it arrives. Terminal-side "local echo" can mask
 * this, but it's not guaranteed on every terminal/monitor, which is
 * exactly the symptom reported: "until I hit enter, it does not show
 * text".
 *
 * Fix: read one character at a time with fgetc() and immediately
 * putchar() it back out ourselves, building the line buffer manually.
 * This also lets us handle backspace/delete properly, which fgets()
 * couldn't do either. This does NOT depend on any terminal setting --
 * it echoes correctly regardless of what serial monitor you use.
 */
static bool read_line(char *buf, size_t bufsize)
{
    size_t len = 0;

    while (1) {
        int c = fgetc(stdin);

        if (c == EOF) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        if (c == '\r' || c == '\n') {
            putchar('\n');
            fflush(stdout);
            buf[len] = '\0';
            return true;
        }

        if (c == 127 || c == 8) { /* backspace / DEL */
            if (len > 0) {
                len--;
                printf("\b \b"); /* erase the character on the terminal too */
                fflush(stdout);
            }
            continue;
        }

        if (c >= 32 && c < 127 && len < bufsize - 1) {
            buf[len++] = (char)c;
            putchar(c); /* echo it back immediately -- this is the fix */
            fflush(stdout);
        }
        /* silently drop anything else (control chars, non-ASCII, or a
         * line that's hit LINE_BUF_SIZE) rather than corrupting the buffer */
    }
}

static void run_repl(void)
{
    const TlmWeights *weights = (const TlmWeights *)g_tlm_weights_blob;
    char line[LINE_BUF_SIZE];
    char response[MAX_GEN_TOKENS + 1]; /* buffered for the OLED preview only --
                                     * serial always gets the full live stream */

    printf("\nKISH-TLM32 ready. Type a story prompt and press Enter.\n> ");
    fflush(stdout);
    oled_waiting();

    while (1) {
        if (!read_line(line, sizeof(line))) {
            continue;
        }

        size_t len = strlen(line);
        if (len == 0) {
            printf("> ");
            fflush(stdout);
            continue;
        }

        oled_prompt(line);

        TlmContext ctx;
        tlm_context_reset(&ctx);
        for (size_t i = 0; i < len; i++) {
            tlm_context_push(&ctx, (unsigned char)line[i]);
        }

        
        float logits[TLM_VOCAB_SIZE];
        int64_t total_us = 0;
        int nonprintable_run = 0;
        int tokens_generated = 0;
        size_t response_len = 0;

        int sentence_endings = 0;

        for (int i = 0; i < MAX_GEN_TOKENS; i++) {
            int64_t t0 = esp_timer_get_time();
            tlm_forward(weights, &ctx, logits);
            int next_tok = tlm_sample(logits, TLM_VOCAB_SIZE, SAMPLE_TEMPERATURE, SAMPLE_TOP_K);
            int64_t t1 = esp_timer_get_time();
            total_us += (t1 - t0);
            tokens_generated++;

            tlm_context_push(&ctx, next_tok);

            int printable = (next_tok >= 32 && next_tok < 127);
            char out_ch = printable ? (char)next_tok : '.';
            putchar(out_ch);
            fflush(stdout);

            if (out_ch == '.' || out_ch == '!' || out_ch == '?')
                sentence_endings++;
            else if (out_ch != ' ' && out_ch != '\n' && out_ch != '\r')
                sentence_endings = 0;

            if (response_len < MAX_GEN_TOKENS) {
                response[response_len++] = out_ch;
            }

            /* Early stop: once the model drifts into a run of unprintable
             * bytes it's almost always signaling "out of learned territory"
             * rather than about to recover -- stop instead of filling the
             * screen with dots. Tune the threshold if you want longer runs. */
            nonprintable_run = printable ? 0 : nonprintable_run + 1;
            if (nonprintable_run >= 20)
                break;

            if (i >= MIN_GEN_TOKENS && sentence_endings >= 2)
                break; /* temporarily raised from 5 for debugging -- lower back once output looks healthy */

            /* watchdog fix + OLED refresh cadence: both happen on the
             * same interval on purpose. Refreshing the OLED every single
             * token would spend I2C time we don't have during inference;
             * every Nth token keeps the progress bar visibly live without
             * meaningfully slowing generation down. */
            if ((i % YIELD_EVERY_N) == (YIELD_EVERY_N - 1)) {
                vTaskDelay(1);
                oled_thinking(i + 1, MAX_GEN_TOKENS);
            }
        }
        response[response_len] = '\0';

        double avg_ms = (double)total_us / tokens_generated / 1000.0;
        printf("\n[%d tokens, avg %.1f ms/token]\n\n> ", tokens_generated, avg_ms);
        fflush(stdout);

        oled_done((float)avg_ms, tokens_generated);
        oled_response_preview(response);
        vTaskDelay(pdMS_TO_TICKS(1500));
        oled_waiting();
    }
}

void app_main(void)
{
    init_console_uart();
    srand((unsigned int)esp_random());

    if (!oled_init()) {
        ESP_LOGW(TAG, "OLED not detected -- continuing on serial only");
    } else {
        oled_boot();
    }

    ESP_LOGI(TAG, "KISH-TLM32 booting — %d params, %d-token context",
             (int)(sizeof(TlmWeights) / sizeof(float)), TLM_CONTEXT_LEN);
    run_repl();
}
