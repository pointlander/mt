// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// mt trains two next-byte models on pg100.txt.
//
// The Markov model counts next-byte contexts of 4, 3, 2, and 1 bytes, and a
// missed lookup uses the next shorter suffix. At byte s, s+4, s+8, ... it
// writes the distribution of the following byte. A single-layer transformer
// with context 1000 reads those distributions and is trained to name the byte
// each distribution was aimed at. Counts are fit on the first 90% of the file.
// Training inputs use leave-one-out counts so a target is not packed into its
// own distribution. Both models are scored on the held-out suffix.
//
// With -prompt, Monte Carlo tree search builds a continuation that maximizes
// the transformer's probability of that string.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"time"
)

const trainFraction = 0.9

func main() {
	steps := flag.Int("steps", 200, "transformer training steps")
	batch := flag.Int("batch", 0, "windows per step (0 uses GOMAXPROCS)")
	lr := flag.Float64("lr", 1e-3, "Adam learning rate")
	seed := flag.Int64("seed", 1, "rng seed")
	evalN := flag.Int("eval", 16, "held-out windows used to score the transformer")
	prompt := flag.String("prompt", "", "generate a continuation of this text")
	genN := flag.Int("gen", 32, "bytes to generate from -prompt")
	sims := flag.Int("sims", 64, "MCTS simulations")
	topk := flag.Int("topk", 16, "MCTS actions kept from each next-byte distribution")
	flag.Parse()
	if *prompt != "" && len(*prompt) < order {
		fmt.Println("prompt must be at least 4 bytes")
		os.Exit(1)
	}

	data, err := os.ReadFile("pg100.txt")
	if err != nil {
		panic(err)
	}
	if *batch < 1 {
		*batch = runtime.GOMAXPROCS(0)
	}
	trainEnd := int(float64(len(data)) * trainFraction)
	fmt.Printf("corpus pg100.txt bytes=%d train=%d test=%d\n", len(data), trainEnd, len(data)-trainEnd)

	fmt.Println("fitting markov counts for orders 4, 3, 2, and 1...")
	markov := newMarkov()
	t0 := time.Now()
	markov.Train(data, trainEnd)
	n4, n3, n2, n1 := markov.Orders()
	fmt.Printf("markov contexts 4=%d 3=%d 2=%d 1=%d observations=%d fit=%s\n",
		n4, n3, n2, n1, markov.observations, time.Since(t0).Round(time.Millisecond))

	trainLOO := markov.Evaluate(data, order, trainEnd, 1, true)
	testAll := markov.Evaluate(data, trainEnd, len(data), 1, false)
	testStride := markov.Evaluate(data, trainEnd, len(data), stride, false)
	fmt.Printf("markov train-loo   n=%d acc=%.4f ce=%.4f bits=%.4f\n", trainLOO.n, trainLOO.acc(), trainLOO.ce(), trainLOO.bits())
	fmt.Printf("markov test        n=%d acc=%.4f ce=%.4f bits=%.4f\n", testAll.n, testAll.acc(), testAll.ce(), testAll.bits())
	fmt.Printf("markov test-stride n=%d acc=%.4f ce=%.4f bits=%.4f\n", testStride.n, testStride.acc(), testStride.ce(), testStride.bits())

	cfg := defaultConfig()
	rng := rand.New(rand.NewSource(*seed))
	weights := newTensors(cfg)
	initTensors(weights, cfg, rng)
	fmt.Printf("transformer layers=1 d=%d heads=%d dff=%d context=%d stride=%d params=%d\n",
		cfg.D, cfg.Heads, cfg.DFF, cfg.Context, stride, paramCount(weights))

	tr0, trCount := countWindows(len(data), 0, trainEnd, cfg.Context, stride)
	te0, teCount := countWindows(len(data), trainEnd, len(data), cfg.Context, stride)
	if trCount < 1 || teCount < 1 {
		panic("not enough text for a context window")
	}
	fmt.Printf("windows train=%d test=%d batch=%d steps=%d\n", trCount, teCount, *batch, *steps)

	testStarts := spreadStarts(te0, teCount, stride, *evalN)
	before := evaluate(weights, cfg, markov, data, testStarts, false)
	fmt.Printf("transformer test before n=%d acc=%.4f ce=%.4f bits=%.4f\n", before.n, before.acc(), before.ce(), before.bits())

	opt := newAdam(weights)
	workers := newWorkers(cfg, *batch)
	trainStart := time.Now()
	var lastLoss float64
	for step := 0; step < *steps; step++ {
		starts := make([]int, *batch)
		for i := range starts {
			starts[i] = tr0 + rng.Intn(trCount)*stride
		}
		loss, acc, gnorm := trainStep(weights, cfg, markov, data, starts, true, workers)
		rate := learningRate(step, 10, *lr)
		clipGrads(workers.grad, 1)
		opt.step(weights, workers.grad, float32(rate))
		lastLoss = loss
		if step%10 == 0 || step+1 == *steps {
			fmt.Printf("step %3d/%d loss=%.4f acc=%.4f grad=%.3f lr=%.2e %s\n",
				step+1, *steps, loss, acc, gnorm, rate, time.Since(trainStart).Round(time.Millisecond))
		}
	}

	after := evaluate(weights, cfg, markov, data, testStarts, false)
	same := markovOnStarts(markov, data, testStarts, cfg.Context, false)
	fmt.Printf("transformer test after  n=%d acc=%.4f ce=%.4f bits=%.4f\n", after.n, after.acc(), after.ce(), after.bits())
	fmt.Printf("markov same windows     n=%d acc=%.4f ce=%.4f bits=%.4f\n", same.n, same.acc(), same.ce(), same.bits())
	fmt.Printf("train last-batch loss=%.4f  held-out ce %.4f -> %.4f (delta %.4f nats)\n",
		lastLoss, before.ce(), after.ce(), before.ce()-after.ce())
	printSample(weights, cfg, markov, data, testStarts[0])

	if *prompt != "" {
		fmt.Printf("mcts prompt=%s gen=%d sims=%d topk=%d\n", strconv.Quote(*prompt), *genN, *sims, *topk)
		tGen := time.Now()
		out := newGenerator(cfg, weights, markov).Search([]byte(*prompt), *genN, *sims, *topk, rng)
		fmt.Printf("greedy logp=%.3f (%.3f bits/byte) %s\n",
			out.greedyLogp, -out.greedyLogp/math.Ln2/float64(*genN), strconv.Quote(string(out.greedy)))
		fmt.Printf("mcts   logp=%.3f (%.3f bits/byte) %s\n",
			out.logp, -out.logp/math.Ln2/float64(*genN), strconv.Quote(string(out.text)))
		fmt.Printf("generated in %s\n", time.Since(tGen).Round(time.Millisecond))
	}

	if testAll.acc() < 0.4 || testAll.ce() > 2.5 {
		fmt.Println("verification failed: markov held-out fit is below the sanity bar")
		os.Exit(1)
	}
	if after.acc() < 0.3 || after.ce() > 4 || !(after.ce() < before.ce()-0.5) {
		fmt.Println("verification failed: transformer did not learn held-out next-byte prediction")
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
			ws:      newWorkspace(cfg, cfg.Context),
			x:       make([]float32, cfg.Context*cfg.Vocab),
			targets: make([]byte, cfg.Context),
		}
	}
	return c
}

func trainStep(w *tensors, cfg Config, m *Markov, data []byte, starts []int, loo bool, c *crew) (loss, acc, gnorm float64) {
	c.grad.zero()
	errc := make(chan error, len(starts))
	type result struct {
		loss    float64
		correct int
	}
	results := make([]result, len(starts))
	for i := range starts {
		i := i
		go func() {
			defer func() {
				if r := recover(); r != nil {
					errc <- fmt.Errorf("%v\n%s", r, debug.Stack())
				}
			}()
			s := &c.slots[i]
			m.Fill(data, starts[i], cfg.Context, loo, s.x, s.targets)
			loss, correct := forwardBackward(cfg, w, s.g, s.ws, s.x, s.targets)
			results[i] = result{loss: loss, correct: correct}
			errc <- nil
		}()
	}
	for range starts {
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
	n := float32(len(starts))
	scaleTensors(c.grad, 1/n)
	loss /= float64(len(starts))
	acc = float64(correct) / float64(len(starts)*cfg.Context)
	return loss, acc, gradNorm(c.grad)
}

func evaluate(w *tensors, cfg Config, m *Markov, data []byte, starts []int, loo bool) score {
	var total score
	ws := newWorkspace(cfg, cfg.Context)
	x := make([]float32, cfg.Context*cfg.Vocab)
	targets := make([]byte, cfg.Context)
	for _, s := range starts {
		m.Fill(data, s, cfg.Context, loo, x, targets)
		loss, correct := forwardBackward(cfg, w, nil, ws, x, targets)
		total.nll += loss * float64(cfg.Context)
		total.correct += correct
		total.n += cfg.Context
	}
	return total
}

func markovOnStarts(m *Markov, data []byte, starts []int, context int, loo bool) score {
	var total score
	var ctx markovKey
	for _, s := range starts {
		for j := 0; j < context; j++ {
			off := s + stride*j
			copy(ctx[:], data[off:off+order])
			tgt := data[off+order]
			p, hit := m.Score(ctx, tgt, loo)
			total.n++
			total.nll += -math.Log(float64(p))
			if hit {
				total.correct++
			}
		}
	}
	return total
}

func printSample(w *tensors, cfg Config, m *Markov, data []byte, start int) {
	ws := newWorkspace(cfg, cfg.Context)
	x := make([]float32, cfg.Context*cfg.Vocab)
	targets := make([]byte, cfg.Context)
	m.Fill(data, start, cfg.Context, false, x, targets)
	forwardBackward(cfg, w, nil, ws, x, targets)
	fmt.Printf("sample test window byte %d (every 4th next-byte)\n", start)
	fmt.Printf("%-10s %-8s %-14s %s\n", "context", "actual", "markov", "transformer")
	var ctx markovKey
	dist := make([]float32, vocab)
	// Show a short interior slice so the window is warmed up.
	for _, j := range []int{100, 101, 102, 103, 104, 200, 201, 202, 400, 401, 800, 801} {
		off := start + stride*j
		copy(ctx[:], data[off:off+order])
		tgt := data[off+order]
		m.Dist(ctx, tgt, false, dist)
		mode := 0
		for i, v := range dist {
			if v > dist[mode] {
				mode = i
			}
		}
		pred, pp := argmaxRow(ws.logits[j*cfg.Vocab : (j+1)*cfg.Vocab])
		fmt.Printf("%-10s %-8s %-14s %s p=%.3f\n",
			strconv.Quote(string(ctx[:])),
			strconv.Quote(string([]byte{tgt})),
			fmt.Sprintf("%s p=%.3f", strconv.Quote(string([]byte{byte(mode)})), dist[tgt]),
			strconv.Quote(string([]byte{byte(pred)})), pp)
	}
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

func spreadStarts(start0, count, step, n int) []int {
	if n > count {
		n = count
	}
	if n < 1 {
		panic("eval windows")
	}
	out := make([]int, n)
	if n == 1 {
		out[0] = start0
		return out
	}
	for i := 0; i < n; i++ {
		idx := int(int64(i) * int64(count-1) / int64(n-1))
		out[i] = start0 + idx*step
	}
	return out
}
