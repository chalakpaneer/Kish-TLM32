#ifndef OLED_H
#define OLED_H

#include <stdbool.h>

/* SSD1306 128x64 I2C OLED driver + KISH-TLM32 status-screen API.
 *
 * Wiring (matches your breadboard):
 *   VCC -> 3V3
 *   GND -> GND
 *   SCL -> GPIO26
 *   SDA -> GPIO25
 *
 * main.c should ONLY ever call the functions below — it never touches
 * I2C or SSD1306 commands directly. That keeps main.c readable and
 * means you can swap this driver (different screen, different pins)
 * without touching the REPL logic at all.
 *
 * IMPORTANT — font limitation: the built-in font only covers uppercase
 * A-Z, digits 0-9, space, and a handful of punctuation ('.', ':', '/',
 * '%', '-', '!', '?', '''). Anything outside that set (lowercase
 * letters, other symbols) is drawn as a blank space rather than
 * garbage. oled_* calls below intentionally uppercase their own status
 * text for this reason. If you want full lowercase support later, swap
 * FONT_5X7 in oled.c for a complete table (e.g. Adafruit's glcdfont.c,
 * MIT licensed) — the rendering code doesn't need to change, just the
 * table and OLED_FONT_FIRST/OLED_FONT_LAST bounds.
 */

/* Call once at boot, before any other oled_* function. Returns false if
 * the display didn't ACK on I2C (check your wiring / address if so) —
 * the REPL should still keep running over serial even if this fails,
 * so treat this as non-fatal. */
bool oled_init(void);

/* Splash screen. Blocks for ~2 seconds. */
void oled_boot(void);

/* Idle screen: shown at REPL startup and after each response. */
void oled_waiting(void);

/* Shown the instant Enter is pressed, before generation starts. */
void oled_prompt(const char *text);

/* Call periodically DURING generation — not every token, every few
 * (main.c should call this every ~4 tokens, matching the serial
 * yield cadence, so the I2C traffic doesn't slow inference down). */
void oled_thinking(int token, int total);

/* Call once generation finishes. */
void oled_done(float ms_per_token, int tokens_generated);

/* Optional: show the first few characters of the actual response
 * underneath the "Done" status. Full response always still goes to
 * serial regardless of what fits here. */
void oled_response_preview(const char *text);

void oled_clear(void);

#endif /* OLED_H */
