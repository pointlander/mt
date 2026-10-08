# mt

`mt` trains a next-byte model on raw text. An order-4 Markov model estimates the distribution of the following byte. A Bayesian model extends that context and is scored on the same task.

The default corpus is [`pg100.txt`](pg100.txt), Project Gutenberg eBook #100, *The Complete Works of William Shakespeare* (5,638,480 bytes).

## Build

Go 1.27.1. The matrix kernels use the experimental `simd` package, so set `GOEXPERIMENT=simd` for every build and test.

```bash
GOEXPERIMENT=simd go test
GOEXPERIMENT=simd go run .
```

The program reads its corpus from the working directory.

## Training

The first 90% of the bytes fit the Markov counts. The last 10% is held out. The Bayesian concentration is chosen on the training span.

```bash
GOEXPERIMENT=simd go run .
```

`-gutenberg N` reads the first N regular `.txt` members of the tar stored in `txt-files.tar.zip`, in archive order, and concatenates them with no separator. That archive is not in this repository. The Markov counts are also fit on `train-v2.0.json` (SQuAD 2.0). Each example is the article title, the paragraph, the question, and the answer. Impossible questions use the answer `unanswerable`. The whole question-answer file is training text. The 90/10 split stays on the books, and the held-out score is still that book suffix.

```bash
GOEXPERIMENT=simd go run . -gutenberg 1000
```

A negative `-gutenberg` value exits 1. `-context` must be positive. A prompt shorter than 4 bytes exits 1. `-temp` must be positive.

After scoring the held-out suffix, the process exits 1 unless the Markov model and the Bayesian model each have accuracy above 0.4 and cross-entropy below 2.5 nats.

## Checkpoints

Training writes the Markov counts before the held-out check.

| Corpus | File |
| --- | --- |
| `pg100.txt` | `markov.bin` |
| `-gutenberg N` | `markov-gN.bin` |

Startup loads that file when it is present and skips the Markov count fit. The extended-context counts and the concentration are built again from the training span. Delete the file to fit the Markov counts again.

## Model

The Markov model stores contexts of 4, 3, 2, and 1 bytes. A missed lookup uses the next shorter suffix. Every bin gets an additive count of 0.01. Training rows scored with leave-one-out do not pack a target into its own input. Held-out evaluation does not use leave-one-out.

The Bayesian model extends the context from 4 bytes to `-context` bytes (8 by default). It counts what followed each context of that length in the training span. The Markov distribution is the mean of a Dirichlet prior over the next byte. The predictive probability is `(kappa * P_markov(y) + count(y)) / (kappa + total)`. A context with no counts reproduces the Markov model. Training scores remove one count of the target. `kappa` is chosen from a fixed grid by the training log loss.

## Generation

Both printed lines are draws from the tempered Bayesian posterior. `-temp` flattens that distribution when it is above 1. The `mcts` line is the last of `-sims` draws. `-sims 0` repeats the sample draw.

```bash
GOEXPERIMENT=simd go run . -prompt 'To be, or not to be' -gen 64 -temp 0.8
```

The printed bits/byte is the negative log probability of the drawn bytes, in bits.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `-context` | 8 | Extended context length in bytes, from 5 to 16 |
| `-seed` | 1 | RNG seed |
| `-prompt` | empty | Generate a continuation of this text |
| `-gen` | 32 | Bytes to generate from `-prompt` |
| `-sims` | 1 | Draws for the `mcts` line; the last one is printed |
| `-temp` | 1 | Temperature of the Bayesian posterior |
| `-gutenberg` | 0 | Fit the Markov counts on the first N books in `txt-files.tar.zip` and on `train-v2.0.json` |

## License

BSD. See [LICENSE](LICENSE).
