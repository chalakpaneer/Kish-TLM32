"""
Trains KISH-TLM32 on a byte-level next-token prediction task and saves a
checkpoint that export_weights.py turns into tlm_weights_data.c.

Usage:
    python train.py --steps 20000
    python train.py --corpus ../datasets/tinystorieskish.txt --steps 20000
    python train.py --demo --steps 500   # tiny built-in corpus, pipeline smoke test

Default corpus is datasets/tinystorieskish.txt (story format).
Use --demo for a quick end-to-end smoke test only.
"""
import argparse
import random
import torch
import torch.nn.functional as F

from tlm_torch import TLM32, CONTEXT_LEN, VOCAB_SIZE

DEMO_CORPUS = """
Theme: Adventure

Title: The Little Robot

Story:
The little robot woke up in the workshop. It blinked its small blue light
and looked around the room. The robot rolled to the window and saw the
garden outside. Birds were singing. The robot decided today was a good day
to explore.
<END>
""".strip()

DEFAULT_CORPUS = "../datasets/tinystorieskish.txt"


def get_batch(data_bytes, batch_size, device):
    xs, ys = [], []
    max_start = len(data_bytes) - CONTEXT_LEN - 1
    for _ in range(batch_size):
        i = random.randint(0, max_start)
        chunk = data_bytes[i:i + CONTEXT_LEN + 1]
        xs.append(list(chunk[:-1]))
        ys.append(list(chunk[1:]))
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="use tiny built-in story corpus (pipeline smoke test)")
    ap.add_argument("--corpus", type=str, default=DEFAULT_CORPUS,
                    help="path to a UTF-8 text file (default: datasets/tinystorieskish.txt)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", type=str, default="tlm32.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.demo:
        data = DEMO_CORPUS.encode("utf-8")
    else:
        with open(args.corpus, "rb") as f:
            data = f.read()

    data = bytes(b for b in data if b < VOCAB_SIZE)  # our vocab is exactly bytes 0..255, always true, kept for clarity
    if len(data) < CONTEXT_LEN + 2:
        raise SystemExit("Corpus too small — needs at least CONTEXT_LEN+2 bytes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TLM32().to(device)
    print(f"KISH-TLM32 parameters: {model.num_params():,}  (device={device})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch_size, device)
        logits = model(x)  # [B, T, VOCAB]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Saved checkpoint to {args.out}")

    # quick sanity-check generation
    model.eval()
    prompt = b"Theme: Adventure\n\nTitle:"
    tokens = list(prompt)[-CONTEXT_LEN:]
    with torch.no_grad():
        for _ in range(60):
            x = torch.tensor([tokens[-CONTEXT_LEN:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            next_tok = int(torch.argmax(logits).item())
            tokens.append(next_tok)
    out_bytes = bytes(t for t in tokens if 32 <= t < 127)  # printable-only, for a readable console preview
    print("Sample generation (printable bytes only):")
    print(out_bytes.decode("ascii", errors="replace"))


if __name__ == "__main__":
    main()
