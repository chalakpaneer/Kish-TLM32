#ifndef TLM_MODEL_H
#define TLM_MODEL_H

#include <stdint.h>
#include "tlm_config.h"

/* Weight layout for ONE transformer block.
 * Every matrix is stored row-major as [in_dim][out_dim], matching
 * tlm_matmul's expected layout. This struct's byte layout MUST match
 * exactly what tools/export_weights.py writes — see that script's
 * header comment before changing anything here.
 */
typedef struct {
    float ln1_w[TLM_EMBED_DIM];
    float ln1_b[TLM_EMBED_DIM];
    float wq[TLM_EMBED_DIM][TLM_EMBED_DIM];
    float wk[TLM_EMBED_DIM][TLM_EMBED_DIM];
    float wv[TLM_EMBED_DIM][TLM_EMBED_DIM];
    float wo[TLM_EMBED_DIM][TLM_EMBED_DIM];
    float ln2_w[TLM_EMBED_DIM];
    float ln2_b[TLM_EMBED_DIM];
    float w1[TLM_EMBED_DIM][TLM_FFN_HIDDEN];
    float b1[TLM_FFN_HIDDEN];
    float w2[TLM_FFN_HIDDEN][TLM_EMBED_DIM];
    float b2[TLM_EMBED_DIM];
} TlmLayerWeights;

typedef struct {
    float token_embedding[TLM_VOCAB_SIZE][TLM_EMBED_DIM];
    float pos_embedding[TLM_CONTEXT_LEN][TLM_EMBED_DIM];
    TlmLayerWeights layers[TLM_N_LAYERS];
    float final_ln_w[TLM_EMBED_DIM];
    float final_ln_b[TLM_EMBED_DIM];
    float output_w[TLM_EMBED_DIM][TLM_VOCAB_SIZE];
    float output_b[TLM_VOCAB_SIZE];
} TlmWeights;

/* Sliding-window token context. */
typedef struct {
    int tokens[TLM_CONTEXT_LEN];
    int len; /* number of valid tokens currently in the window, 0..TLM_CONTEXT_LEN */
} TlmContext;

void tlm_context_reset(TlmContext *ctx);
void tlm_context_push(TlmContext *ctx, int token);

/* Runs the full forward pass over ctx->tokens[0..ctx->len) and writes
 * next-token logits (TLM_VOCAB_SIZE floats) into logits_out.
 * No KV-cache — recomputes from scratch every call. Fine at this scale
 * (context <= 16, embed = 32); revisit if you grow the model later.
 */
void tlm_forward(const TlmWeights *w, const TlmContext *ctx, float *logits_out);

#endif /* TLM_MODEL_H */
