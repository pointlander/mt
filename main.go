// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// mt trains a next-byte model on pg100.txt.
//
// The Markov model counts next-byte contexts of 4, 3, 2, and 1 bytes, and a
// missed lookup uses the next shorter suffix. A Bayesian model extends that
// context. The Markov distribution is the mean of a Dirichlet prior, and the
// bytes in a longer window are the observations. The posterior predictive is
// the next-byte distribution. Counts are fit on the first 90% of the file.
// The prior strength is chosen on that span. Both models are scored on the
// held-out suffix.
// If markov.bin is present it is loaded and the counts are not fit again.
// A training run writes that file.
// -gutenberg N reads the first N books in txt-files.tar.zip, fits the counts
// on them and on train-v2.0.json, and uses markov-gN.bin the same way.
//
// With -prompt, the sample line and the MCTS line are both random draws from
// the tempered Bayesian posterior. -temp scales that distribution. The MCTS
// line is the last of -sims draws.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime/debug"
	"strconv"
	"time"
)

const trainFraction = 0.9

func main() {
	seed := flag.Int64("seed", 1, "rng seed")
	prompt := flag.String("prompt", "", "generate a continuation of this text")
	genN := flag.Int("gen", 32, "bytes to generate from -prompt")
	sims := flag.Int("sims", 1, "softmax samples for the MCTS line; the last one is printed")
	temp := flag.Float64("temp", 1, "softmax temperature for generation")
	contextN := flag.Int("context", 8, "extended context length in bytes")
	gutenberg := flag.Int("gutenberg", 0, "fit the Markov counts on the first N books in txt-files.tar.zip and on train-v2.0.json")
	flag.Parse()
	if *prompt != "" && len(*prompt) < order {
		fmt.Println("prompt must be at least 4 bytes")
		os.Exit(1)
	}
	if !(*temp > 0) {
		fmt.Println("temperature must be positive")
		os.Exit(1)
	}
	if *gutenberg < 0 {
		fmt.Println("gutenberg count must be non-negative")
		os.Exit(1)
	}

	var (
		data []byte
		err  error
		read time.Duration
	)
	if *gutenberg > 0 {
		tRead := time.Now()
		data, err = gutenbergBooks("txt-files.tar.zip", *gutenberg)
		read = time.Since(tRead)
	} else {
		data, err = os.ReadFile("pg100.txt")
	}
	if err != nil {
		panic(err)
	}
	if *contextN <= order || *contextN > 16 {
		fmt.Println("context must be from 5 to 16 bytes")
		os.Exit(1)
	}
	trainEnd := int(float64(len(data)) * trainFraction)
	if *gutenberg > 0 {
		fmt.Printf("corpus txt-files.tar.zip books=%d bytes=%d train=%d test=%d %s\n",
			*gutenberg, len(data), trainEnd, len(data)-trainEnd, read.Round(time.Millisecond))
	} else {
		fmt.Printf("corpus pg100.txt bytes=%d train=%d test=%d\n", len(data), trainEnd, len(data)-trainEnd)
	}

	mFile, _ := checkpointNames(*gutenberg)
	haveMarkov, err := fileExists(mFile)
	if err != nil {
		panic(err)
	}

	var markov *Markov
	if haveMarkov {
		t0 := time.Now()
		markov, err = loadMarkov(mFile)
		if err != nil {
			panic(err)
		}
		n4, n3, n2, n1 := markov.Orders()
		fmt.Printf("loaded %s contexts 4=%d 3=%d 2=%d 1=%d observations=%d %s\n",
			mFile, n4, n3, n2, n1, markov.observations, time.Since(t0).Round(time.Millisecond))
	} else {
		fmt.Println("fitting markov counts for orders 4, 3, 2, and 1...")
		markov = newMarkov()
		t0 := time.Now()
		markov.Train(data, trainEnd)
		if *gutenberg > 0 {
			tQA := time.Now()
			qa, nQA, err := squadTrainText(squadFile)
			if err != nil {
				panic(err)
			}
			fmt.Printf("corpus %s examples=%d bytes=%d %s\n",
				squadFile, nQA, len(qa), time.Since(tQA).Round(time.Millisecond))
			markov.Train(qa, len(qa))
		}
		n4, n3, n2, n1 := markov.Orders()
		fmt.Printf("markov contexts 4=%d 3=%d 2=%d 1=%d observations=%d fit=%s\n",
			n4, n3, n2, n1, markov.observations, time.Since(t0).Round(time.Millisecond))
		if err = saveMarkov(mFile, markov); err != nil {
			panic(err)
		}
		fmt.Printf("saved %s\n", mFile)
	}

	trainLOO := markov.Evaluate(data, order, trainEnd, 1, true)
	testAll := markov.Evaluate(data, trainEnd, len(data), 1, false)
	testStride := markov.Evaluate(data, trainEnd, len(data), stride, false)
	fmt.Printf("markov train-loo   n=%d acc=%.4f ce=%.4f bits=%.4f\n", trainLOO.n, trainLOO.acc(), trainLOO.ce(), trainLOO.bits())
	fmt.Printf("markov test        n=%d acc=%.4f ce=%.4f bits=%.4f\n", testAll.n, testAll.acc(), testAll.ce(), testAll.bits())
	fmt.Printf("markov test-stride n=%d acc=%.4f ce=%.4f bits=%.4f\n", testStride.n, testStride.acc(), testStride.ce(), testStride.bits())

	fmt.Println("fitting bayesian concentration on the training span...")
	tBayes := time.Now()
	bayes, trainBayes := fitBayes(markov, data, trainEnd, *contextN)
	fmt.Printf("bayes context=%d kappa=%g contexts=%d train-loo n=%d ce=%.4f bits=%.4f fit=%s\n",
		bayes.nctx, bayes.kappa, len(bayes.tab), trainBayes.n, trainBayes.ce(), trainBayes.bits(), time.Since(tBayes).Round(time.Millisecond))
	held := bayes.Evaluate(data, trainEnd, len(data), false)
	fmt.Printf("bayes test         n=%d acc=%.4f ce=%.4f bits=%.4f\n", held.n, held.acc(), held.ce(), held.bits())
	fmt.Printf("held-out ce bayes %.4f  markov %.4f (delta %.4f nats)\n",
		held.ce(), testAll.ce(), testAll.ce()-held.ce())

	if *prompt != "" {
		rng := rand.New(rand.NewSource(*seed))
		fmt.Printf("sample prompt=%s gen=%d sims=%d temp=%g\n", strconv.Quote(*prompt), *genN, *sims, *temp)
		tGen := time.Now()
		out := searchContinuation([]byte(*prompt), *genN, *sims, rng, func(prefix []byte) []float32 {
			return temperDist(bayes.distribution(prefix), *temp)
		})
		fmt.Printf("sample logp=%.3f (%.3f bits/byte) %s\n",
			out.sampleLogp, -out.sampleLogp/math.Ln2/float64(*genN), strconv.Quote(string(out.sample)))
		fmt.Printf("mcts   logp=%.3f (%.3f bits/byte) %s\n",
			out.logp, -out.logp/math.Ln2/float64(*genN), strconv.Quote(string(out.text)))
		fmt.Printf("generated in %s\n", time.Since(tGen).Round(time.Millisecond))
	}

	if testAll.acc() < 0.4 || testAll.ce() > 2.5 {
		fmt.Println("verification failed: markov held-out fit is below the sanity bar")
		os.Exit(1)
	}
	if held.acc() < 0.4 || held.ce() > 2.5 {
		fmt.Println("verification failed: bayesian held-out fit is below the sanity bar")
		os.Exit(1)
	}
	fmt.Println("verified")
}

type crew struct {
	slots []slot
	grad  *tensors
}

type slot struct {
	g       *tensors
	ws      *workspace
	x       []float32
	targets []byte
}

func newWorkers(cfg Config, n int) *crew {
	c := &crew{grad: newTensors(cfg), slots: make([]slot, n)}
	for i := range c.slots {
		c.slots[i] = slot{
			g:       newTensors(cfg),
			ws:      newWorkspace(cfg, cfg.seqLen()),
			x:       make([]float32, cfg.seqLen()*cfg.Vocab),
			targets: make([]byte, cfg.seqLen()),
		}
	}
	return c
}

func forwardSlots(w *tensors, cfg Config, c *crew, n int) (loss, acc, gnorm float64) {
	c.grad.zero()
	errc := make(chan error, n)
	type result struct {
		loss    float64
		correct int
	}
	results := make([]result, n)
	for i := 0; i < n; i++ {
		i := i
		go func() {
			defer func() {
				if r := recover(); r != nil {
					errc <- fmt.Errorf("%v\n%s", r, debug.Stack())
				}
			}()
			s := &c.slots[i]
			loss, correct := forwardBackward(cfg, w, s.g, s.ws, s.x, s.targets)
			results[i] = result{loss: loss, correct: correct}
			errc <- nil
		}()
	}
	for range n {
		if err := <-errc; err != nil {
			panic(err)
		}
	}
	var correct int
	for i := range results {
		loss += results[i].loss
		correct += results[i].correct
		addTensors(c.grad, c.slots[i].g)
	}
	scaleTensors(c.grad, 1/float32(n))
	loss /= float64(n)
	acc = float64(correct) / float64(n*cfg.Context)
	return loss, acc, gradNorm(c.grad)
}

// countWindows returns the first byte offset and the number of stride-aligned
// windows whose context length targets all lie in [lo, hi).
func countWindows(n, lo, hi, context, step int) (start0, count int) {
	if hi > n {
		hi = n
	}
	if context < 1 || step < 1 || lo >= hi {
		return 0, 0
	}
	raw := lo - step
	if raw < 0 {
		raw = 0
	}
	start0 = alignUp(raw, step)
	maxS := hi - 1 - step*context
	limit := n - 1 - step*context
	if limit < maxS {
		maxS = limit
	}
	if maxS < start0 {
		return 0, 0
	}
	sMax := maxS - maxS%step
	if sMax < start0 {
		return 0, 0
	}
	return start0, (sMax-start0)/step + 1
}

func alignUp(x, m int) int {
	if x <= 0 {
		return 0
	}
	if r := x % m; r != 0 {
		return x + (m - r)
	}
	return x
}
