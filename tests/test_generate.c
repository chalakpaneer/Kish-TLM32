/* Host test harness — compiles the SAME tlm_math.c / tlm_model.c that run on
 * the ESP32, but with plain gcc, so you can sanity-check training + export
 * before ever touching hardware.
 *
 * Build:  see tests/Makefile
 * Usage:  ./tlm32_host "The robot" 80 [temperature] [top_k]
 *         temperature 0 (default) = deterministic argmax, matches old behavior
 *         temperature >0, e.g. 0.8, with top_k e.g. 10 = sampled output
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tlm_config.h"
#include "tlm_math.h"
#include "tlm_model.h"
#include "tlm_weights_data.h"

int main(int argc, char **argv)
{
    const char *prompt = argc > 1 ? argv[1] : "The robot";
    int gen_len = argc > 2 ? atoi(argv[2]) : 80;
    float temperature = argc > 3 ? (float)atof(argv[3]) : 0.0f;
    int top_k = argc > 4 ? atoi(argv[4]) : 0;
    srand((unsigned int)time(NULL));

    printf("Loaded weight blob: %u bytes (expected %zu)\n",
           g_tlm_weights_blob_len, sizeof(TlmWeights));
    if (g_tlm_weights_blob_len != (unsigned)sizeof(TlmWeights)) {
        fprintf(stderr,
            "ERROR: blob size does not match TlmWeights struct size.\n"
            "Did tlm_config.h change after export_weights.py was last run?\n");
        return 1;
    }

    const TlmWeights *w = (const TlmWeights *)g_tlm_weights_blob;

    TlmContext ctx;
    tlm_context_reset(&ctx);
    size_t plen = strlen(prompt);
    for (size_t i = 0; i < plen; i++) {
        tlm_context_push(&ctx, (unsigned char)prompt[i]);
    }

    printf("%s", prompt);
    float logits[TLM_VOCAB_SIZE];
    int nonprintable_run = 0;
    for (int i = 0; i < gen_len; i++) {
        tlm_forward(w, &ctx, logits);
        int next_tok = tlm_sample(logits, TLM_VOCAB_SIZE, temperature, top_k);
        tlm_context_push(&ctx, next_tok);
        int printable = (next_tok >= 32 && next_tok < 127);
        putchar(printable ? next_tok : '.');
        nonprintable_run = printable ? 0 : nonprintable_run + 1;
        if (nonprintable_run >= 5) break; /* matches main.c's early-stop behavior */
    }
    printf("\n");
    return 0;
}
