#include <math.h>
#include <stdlib.h>
#include "tlm_math.h"
#include "tlm_config.h"

void tlm_matmul(const float *x, const float *w, float *out, int in_dim, int out_dim)
{
    for (int o = 0; o < out_dim; o++) {
        float acc = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            acc += x[i] * w[i * out_dim + o];
        }
        out[o] = acc;
    }
}

void tlm_add_bias(float *out, const float *b, int dim)
{
    for (int i = 0; i < dim; i++) {
        out[i] += b[i];
    }
}

void tlm_layernorm(const float *x, const float *w, const float *b, float *out, int dim)
{
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;

    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)dim;

    float inv_std = 1.0f / sqrtf(var + TLM_LN_EPS);
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
    }
}

void tlm_softmax(float *x, int n)
{
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

void tlm_relu(float *x, int n)
{
    for (int i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

void tlm_add_inplace(float *dst, const float *src, int n)
{
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

int tlm_argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

int tlm_sample(float *logits, int n, float temperature, int top_k)
{
    if (temperature <= 0.0f) {
        return tlm_argmax(logits, n);
    }

    for (int i = 0; i < n; i++) logits[i] /= temperature;
    tlm_softmax(logits, n);

    if (top_k > 0 && top_k < n) {
        /* zero out every probability except the top_k largest.
         * n is small (<=256) so a simple O(n*top_k) selection is fine. */
        int kept[64]; /* top_k is expected to be small; guard against misuse */
        if (top_k > 64) top_k = 64;
        int kept_count = 0;
        float work[TLM_VOCAB_SIZE];
        for (int i = 0; i < n; i++) work[i] = logits[i];

        for (int k = 0; k < top_k; k++) {
            int best = -1;
            for (int i = 0; i < n; i++) {
                if (work[i] < 0.0f) continue; /* already selected, marked used */
                if (best == -1 || work[i] > work[best]) best = i;
            }
            if (best == -1) break;
            kept[kept_count++] = best;
            work[best] = -1.0f; /* mark used */
        }

        float kept_sum = 0.0f;
        for (int i = 0; i < n; i++) {
            int is_kept = 0;
            for (int k = 0; k < kept_count; k++) if (kept[k] == i) { is_kept = 1; break; }
            if (!is_kept) logits[i] = 0.0f;
            else kept_sum += logits[i];
        }
        if (kept_sum > 0.0f) {
            float inv = 1.0f / kept_sum;
            for (int i = 0; i < n; i++) logits[i] *= inv;
        }
    }

    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += logits[i];
        if (r < acc) return i;
    }
    return n - 1; /* fallback for float rounding */
}
