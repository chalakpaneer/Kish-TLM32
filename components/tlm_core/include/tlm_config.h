#ifndef TLM_CONFIG_H
#define TLM_CONFIG_H

/* ---- TLM32 hyperparameters ----
 * Keep this in sync with tools/train.py — the exported weight blob's
 * layout depends on every one of these numbers.
 */
#define TLM_VOCAB_SIZE   256   /* byte-level tokenizer: token id == raw byte value */
#define TLM_CONTEXT_LEN   64   /* max tokens of context (sliding window)           */
#define TLM_EMBED_DIM     32   /* residual stream width                           */
#define TLM_N_HEADS        2   /* attention heads                                 */
#define TLM_HEAD_DIM     (TLM_EMBED_DIM / TLM_N_HEADS)
#define TLM_N_LAYERS       2   /* transformer blocks                              */
#define TLM_FFN_HIDDEN    64   /* feed-forward inner dim                          */
#define TLM_LN_EPS     1e-5f

#endif /* TLM_CONFIG_H */
