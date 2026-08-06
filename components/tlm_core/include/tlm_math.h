#ifndef TLM_MATH_H
#define TLM_MATH_H

/* out[o] = sum_i x[i] * w[i*out_dim + o]   (w stored row-major as [in_dim][out_dim]) */
void tlm_matmul(const float *x, const float *w, float *out, int in_dim, int out_dim);

/* out[o] += b[o] for o in [0, dim) */
void tlm_add_bias(float *out, const float *b, int dim);

/* Standard LayerNorm with learned scale/shift, eps = TLM_LN_EPS */
void tlm_layernorm(const float *x, const float *w, const float *b, float *out, int dim);

/* In-place softmax over n elements */
void tlm_softmax(float *x, int n);

/* In-place ReLU over n elements */
void tlm_relu(float *x, int n);

/* dst[i] += src[i] for i in [0, n) — used for residual connections */
void tlm_add_inplace(float *dst, const float *src, int n);

/* index of the largest element in x[0..n) */
int tlm_argmax(const float *x, int n);

/* Samples a token from logits[0..n).
 * temperature <= 0.0f  -> deterministic argmax (old behavior).
 * temperature  > 0.0f  -> scales logits by 1/temperature, softmaxes,
 *                         restricts to the top_k highest-probability
 *                         tokens (top_k <= 0 or >= n means "no restriction"),
 *                         then samples from that renormalized distribution.
 * NOTE: destructively modifies logits in place (softmaxes it).
 * Caller must seed the RNG once at startup (srand()); this uses rand().
 */
int tlm_sample(float *logits, int n, float temperature, int top_k);

#endif /* TLM_MATH_H */
