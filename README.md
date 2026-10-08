# mt

`mt` trains two next-byte models on raw text. An order-4 Markov model estimates the distribution of the following byte. A single-layer transformer reads those distributions and is scored on the same task.

The default corpus is [`pg100.txt`](pg100.txt), Project Gutenberg eBook #100, *The Complete Works of William Shakespeare* (5,638,480 bytes).

## Build

Go 1.27.1. The matrix kernels use the experimental `simd` package, so set `GOEXPERIMENT=simd` for every build and test.

```bash
GOEXPERIMENT=simd go test
GOEXPERIMENT=simd go run .
```

The program reads its corpus from the working directory.

## Training

The first 90% of the bytes fit the Markov counts and train the transformer. The last 10% is held out.

A default run draws `-steps` batches of random windows from the training span. `-batch 0` uses `GOMAXPROCS`. The learning rate warms up over the first 10 steps, then stays at `-lr`. Gradients are clipped to norm 1.

```bash
GOEXPERIMENT=simd go run . -steps 200 -lr 1e-3
```

`-gutenberg N` reads the first N regular `.txt` members of the tar stored in `txt-files.tar.zip`, in archive order, and concatenates them with no separator. That archive is not in this repository. The run is one pass over non-overlapping windows. With the default context and group of 1000, each stride-aligned training position is a target once. `-steps` does not set the length of that pass. A thousand books is a few hours.

The same pass then trains on `train-v2.0.json` (SQuAD 2.0). Each example is the article title, the paragraph, the question, and the answer. Impossible questions use the answer `unanswerable`. The whole file is training text. The 90/10 split stays on the books, and the held-out score is still that book suffix.

```bash
GOEXPERIMENT=simd go run . -gutenberg 1000
```

A negative `-gutenberg` value exits 1. A prompt shorter than 4 bytes exits 1. `-temp` must be positive.

After scoring the held-out suffix, the process exits 1 unless both of these hold:

- Markov accuracy is above 0.4 and cross-entropy is below 2.5 nats.
- Transformer accuracy is above 0.3 and cross-entropy is below 4 nats. A process that trained in this run must also improve held-out cross-entropy by more than 0.5 nats.

## Checkpoints

Training writes both files after the training loop and before the final held-out check.

| Corpus | Files |
| --- | --- |
| `pg100.txt` | `markov.bin`, `weights.bin` |
| `-gutenberg N` | `markov-gN.bin`, `weights-gN.bin` |

Startup loads the pair when both files are present and skips fitting, initialization, and the training loop. The 0.5-nat improvement check is skipped with them. The absolute bars still run. One file without the other stops the program. Delete the pair to train again.

A loaded `pg100.txt` run still rebuilds the distribution prefix from the corpus. That prefix is about 2.7 GiB. A loaded Gutenberg run does not rebuild its group sums.

## Model

The Markov model stores contexts of 4, 3, 2, and 1 bytes. A missed lookup uses the next shorter suffix. Every bin gets an additive count of 0.01. Rows are written every 4 bytes, so successive inputs are disjoint 4-grams. The scored training rows use leave-one-out counts, so a target is not packed into its own input. History sums and evaluation do not.

The transformer is one pre-norm causal layer: width 64, 4 heads, feed-forward width 256, 211,200 parameters. Each position consumes a 256-dimensional distribution. The sequence is 2000 vectors. The first 1000 are sums of 1000 earlier rows, so the previous 1,000,000 rows are folded into the context. The last 1000 are the raw rows, and the loss is the mean next-byte cross-entropy on those rows only.

On `pg100.txt` the sums come from an exclusive prefix of the raw rows. On the Gutenberg path each bucket is copied from a table of group sums, about 109 MB for the first 1000 books.

## Generation

Both printed lines are draws from the tempered softmax of the transformer. `-temp` divides the logits. The `mcts` line is the last of `-sims` draws. `-sims 0` repeats the sample draw.

```bash
GOEXPERIMENT=simd go run . -prompt 'To be, or not to be' -gen 64 -temp 0.8
```

The printed bits/byte is the negative log probability of the drawn bytes, in bits.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `-steps` | 200 | Transformer training steps on `pg100.txt` |
| `-batch` | 0 | Windows per step; 0 uses `GOMAXPROCS` |
| `-lr` | 1e-3 | Adam learning rate |
| `-seed` | 1 | RNG seed |
| `-eval` | 16 | Held-out windows used to score the transformer |
| `-prompt` | empty | Generate a continuation of this text |
| `-gen` | 32 | Bytes to generate from `-prompt` |
| `-sims` | 1 | Softmax samples for the `mcts` line; the last one is printed |
| `-temp` | 1 | Softmax temperature for generation |
| `-gutenberg` | 0 | One-shot train on the first N books in `txt-files.tar.zip` and on `train-v2.0.json` |

## License

BSD. See [LICENSE](LICENSE).
