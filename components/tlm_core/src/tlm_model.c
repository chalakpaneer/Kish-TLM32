#include <string.h>
#include <math.h>
#include "tlm_model.h"
#include "tlm_math.h"

void tlm_context_reset(TlmContext *ctx)
{
    ctx->len = 0;
}

void tlm_context_push(TlmContext *ctx, int token)
{
    if (ctx->len < TLM_CONTEXT_LEN) {
        ctx->tokens[ctx->len++] = token;
    } else {
        /* slide the window left by one, drop oldest token */
        memmove(ctx->tokens, ctx->tokens + 1, (TLM_CONTEXT_LEN - 1) * sizeof(int));
        ctx->tokens[TLM_CONTEXT_LEN - 1] = token;
    }
}

/* Work buffers. Static, not on the stack — this runs on an ESP32 with a
 * small task stack, and re-entrancy isn't needed (single inference loop). */
static float s_hidden[TLM_CONTEXT_LEN][TLM_EMBED_DIM];   /* residual stream per position */
static float s_normed[TLM_EMBED_DIM];
static float s_q[TLM_CONTEXT_LEN][TLM_EMBED_DIM];
static float s_k[TLM_CONTEXT_LEN][TLM_EMBED_DIM];
static float s_v[TLM_CONTEXT_LEN][TLM_EMBED_DIM];
static float s_attn_out[TLM_EMBED_DIM];
static float s_scores[TLM_CONTEXT_LEN];
static float s_ffn_hidden[TLM_FFN_HIDDEN];
static float s_ffn_out[TLM_EMBED_DIM];

static void self_attention(const TlmLayerWeights *lw, int n)
{
    /* project every position to q/k/v (each still using the *unnormed*
     * s_normed buffer computed per-position by the caller) */
    for (int t = 0; t < n; t++) {
        tlm_layernorm(s_hidden[t], lw->ln1_w, lw->ln1_b, s_normed, TLM_EMBED_DIM);
        tlm_matmul(s_normed, (const float *)lw->wq, s_q[t], TLM_EMBED_DIM, TLM_EMBED_DIM);
        tlm_matmul(s_normed, (const float *)lw->wk, s_k[t], TLM_EMBED_DIM, TLM_EMBED_DIM);
        tlm_matmul(s_normed, (const float *)lw->wv, s_v[t], TLM_EMBED_DIM, TLM_EMBED_DIM);
    }

    float scale = 1.0f / sqrtf((float)TLM_HEAD_DIM);

    for (int t = 0; t < n; t++) {
        float concat[TLM_EMBED_DIM];

        for (int h = 0; h < TLM_N_HEADS; h++) {
            int off = h * TLM_HEAD_DIM;

            /* causal: query at position t only attends to keys 0..t */
            for (int j = 0; j <= t; j++) {
                float dot = 0.0f;
                for (int d = 0; d < TLM_HEAD_DIM; d++) {
                    dot += s_q[t][off + d] * s_k[j][off + d];
                }
                s_scores[j] = dot * scale;
            }
            tlm_softmax(s_scores, t + 1);

            for (int d = 0; d < TLM_HEAD_DIM; d++) {
                float acc = 0.0f;
                for (int j = 0; j <= t; j++) {
                    acc += s_scores[j] * s_v[j][off + d];
                }
                concat[off + d] = acc;
            }
        }

        tlm_matmul(concat, (const float *)lw->wo, s_attn_out, TLM_EMBED_DIM, TLM_EMBED_DIM);
        tlm_add_inplace(s_hidden[t], s_attn_out, TLM_EMBED_DIM);
    }
}

static void feed_forward(const TlmLayerWeights *lw, int n)
{
    for (int t = 0; t < n; t++) {
        tlm_layernorm(s_hidden[t], lw->ln2_w, lw->ln2_b, s_normed, TLM_EMBED_DIM);
        tlm_matmul(s_normed, (const float *)lw->w1, s_ffn_hidden, TLM_EMBED_DIM, TLM_FFN_HIDDEN);
        tlm_add_bias(s_ffn_hidden, lw->b1, TLM_FFN_HIDDEN);
        tlm_relu(s_ffn_hidden, TLM_FFN_HIDDEN);
        tlm_matmul(s_ffn_hidden, (const float *)lw->w2, s_ffn_out, TLM_FFN_HIDDEN, TLM_EMBED_DIM);
        tlm_add_bias(s_ffn_out, lw->b2, TLM_EMBED_DIM);
        tlm_add_inplace(s_hidden[t], s_ffn_out, TLM_EMBED_DIM);
    }
}

void tlm_forward(const TlmWeights *w, const TlmContext *ctx, float *logits_out)
{
    int n = ctx->len;
    if (n == 0) {
        for (int i = 0; i < TLM_VOCAB_SIZE; i++) logits_out[i] = 0.0f;
        return;
    }

    for (int t = 0; t < n; t++) {
        memcpy(s_hidden[t], w->token_embedding[ctx->tokens[t]], sizeof(float) * TLM_EMBED_DIM);
        tlm_add_inplace(s_hidden[t], w->pos_embedding[t], TLM_EMBED_DIM);
    }

    for (int l = 0; l < TLM_N_LAYERS; l++) {
        self_attention(&w->layers[l], n);
        feed_forward(&w->layers[l], n);
    }

    /* only the last position's logits are needed for next-token prediction */
    int last = n - 1;
    tlm_layernorm(s_hidden[last], w->final_ln_w, w->final_ln_b, s_normed, TLM_EMBED_DIM);
    tlm_matmul(s_normed, (const float *)w->output_w, logits_out, TLM_EMBED_DIM, TLM_VOCAB_SIZE);
    tlm_add_bias(logits_out, w->output_b, TLM_VOCAB_SIZE);
}
