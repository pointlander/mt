// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// mt trains two next-byte models on pg100.txt.
//
// The Markov model counts next-byte contexts of 4, 3, 2, and 1 bytes, and a
// missed lookup uses the next shorter suffix. At byte s, s+4, s+8, ... it
// writes the distribution of the following byte. A single-layer transformer
// reads 2000 of those vectors. The first 1000 are sums of 1000 earlier
// vectors each, folding the previous 1,000,000 vectors into the context.
// The last 1000 are the raw distributions and are the ones that are scored.
// Counts are fit on the first 90% of the file. Training inputs for those
// scored rows use leave-one-out counts so a target is not packed into its
// own distribution. Both models are scored on the held-out suffix.
// If markov.bin and weights.bin are both present they are loaded and
// training is skipped. A training run writes both files.
// -gutenberg N reads the first N books in txt-files.tar.zip, trains one
// pass over them, and uses markov-gN.bin and weights-gN.bin the same way.
//
// With -prompt, the sample line and the MCTS line are both random draws from
// the tempered softmax. -temp scales that softmax. The MCTS line is the last
// of -sims draws.
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
	sims := flag.Int("sims", 1, "softmax samples for the MCTS line; the last one is printed")
	temp := flag.Float64("temp", 1, "softmax temperature for generation")
	gutenberg := flag.Int("gutenberg", 0, "one-shot train on the first N books in txt-files.tar.zip")
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
	if *batch < 1 {
		*batch = runtime.GOMAXPROCS(0)
	}
	trainEnd := int(float64(len(data)) * trainFraction)
	if *gutenberg > 0 {
		fmt.Printf("corpus txt-files.tar.zip books=%d bytes=%d train=%d test=%d %s\n",
			*gutenberg, len(data), trainEnd, len(data)-trainEnd, read.Round(time.Millisecond))
	} else {
		fmt.Printf("corpus pg100.txt bytes=%d train=%d test=%d\n", len(data), trainEnd, len(data)-trainEnd)
	}

	cfg := defaultConfig()
	rng := rand.New(rand.NewSource(*seed))
	mFile, wFile := checkpointNames(*gutenberg)
	haveMarkov, err := fileExists(mFile)
	if err != nil {
		panic(err)
	}
	haveWeights, err := fileExists(wFile)
	if err != nil {
		panic(err)
	}
	if haveMarkov != haveWeights {
		fmt.Printf("need both %s and %s, or neither\n", mFile, wFile)
		os.Exit(1)
	}

	var markov *Markov
	var weights *tensors
	loaded := haveMarkov && haveWeights
	if loaded {
		t0 := time.Now()
		markov, err = loadMarkov(mFile)
		if err != nil {
			panic(err)
		}
		n4, n3, n2, n1 := markov.Orders()
		fmt.Printf("loaded %s contexts 4=%d 3=%d 2=%d 1=%d observations=%d %s\n",
			mFile, n4, n3, n2, n1, markov.observations, time.Since(t0).Round(time.Millisecond))
		t1 := time.Now()
		weights, err = loadWeights(wFile, cfg)
		if err != nil {
			panic(err)
		}
		fmt.Printf("loaded %s params=%d %s\n", wFile, paramCount(weights), time.Since(t1).Round(time.Millisecond))
	} else {
		fmt.Println("fitting markov counts for orders 4, 3, 2, and 1...")
		markov = newMarkov()
		t0 := time.Now()
		markov.Train(data, trainEnd)
		n4, n3, n2, n1 := markov.Orders()
		fmt.Printf("markov contexts 4=%d 3=%d 2=%d 1=%d observations=%d fit=%s\n",
			n4, n3, n2, n1, markov.observations, time.Since(t0).Round(time.Millisecond))
		weights = newTensors(cfg)
		initTensors(weights, cfg, rng)
	}

	trainLOO := markov.Evaluate(data, order, trainEnd, 1, true)
	testAll := markov.Evaluate(data, trainEnd, len(data), 1, false)
	testStride := markov.Evaluate(data, trainEnd, len(data), stride, false)
	fmt.Printf("markov train-loo   n=%d acc=%.4f ce=%.4f bits=%.4f\n", trainLOO.n, trainLOO.acc(), trainLOO.ce(), trainLOO.bits())
	fmt.Printf("markov test        n=%d acc=%.4f ce=%.4f bits=%.4f\n", testAll.n, testAll.acc(), testAll.ce(), testAll.bits())
	fmt.Printf("markov test-stride n=%d acc=%.4f ce=%.4f bits=%.4f\n", testStride.n, testStride.acc(), testStride.ce(), testStride.bits())

	var pre *distPrefix
	var sums []float32
	if *gutenberg > 0 {
		if !loaded {
			fmt.Println("summing markov distributions into groups...")
			tPre := time.Now()
			sums = buildGroupSums(markov, data, cfg.Group)
			nvec := 0
			if len(data) > order {
				nvec = (len(data) - order) / stride
			}
			ng := len(sums) / vocab
			fmt.Printf("group sums groups=%d vectors=%d bytes=%d fit=%s\n",
				ng, nvec, len(sums)*4, time.Since(tPre).Round(time.Millisecond))
		}
	} else {
		fmt.Println("summing markov distributions for the compressed context...")
		tPre := time.Now()
		pre = markov.buildDistPrefix(data)
		fmt.Printf("distribution prefix vectors=%d bytes=%d fit=%s\n",
			pre.n, len(pre.sum)*8, time.Since(tPre).Round(time.Millisecond))
	}

	fmt.Printf("transformer layers=1 d=%d heads=%d dff=%d context=%d compressed=%d group=%d stride=%d params=%d\n",
		cfg.D, cfg.Heads, cfg.DFF, cfg.Context, cfg.Compressed, cfg.Group, stride, paramCount(weights))

	tr0, trCount := countWindows(len(data), 0, trainEnd, cfg.Context, stride)
	te0, teCount := countWindows(len(data), trainEnd, len(data), cfg.Context, stride)
	if trCount < 1 || teCount < 1 {
		panic("not enough text for a context window")
	}
	var shot []int
	if *gutenberg > 0 {
		shot = oneShotStarts(len(data), trainEnd, cfg.Context, cfg.Group, stride)
		fmt.Printf("windows train=%d test=%d batch=%d one-shot=%d\n", trCount, teCount, *batch, len(shot))
	} else {
		fmt.Printf("windows train=%d test=%d batch=%d steps=%d\n", trCount, teCount, *batch, *steps)
	}

	testStarts := spreadStarts(te0, teCount, stride, *evalN)
	var before score
	var lastLoss float64
	if !loaded {
		before = evaluate(weights, cfg, markov, pre, data, testStarts, false)
		fmt.Printf("transformer test before n=%d acc=%.4f ce=%.4f bits=%.4f\n", before.n, before.acc(), before.ce(), before.bits())

		if *gutenberg > 0 {
			lastLoss = oneShotTrain(weights, cfg, markov, sums, data, shot, *batch, *lr)
		} else {
			opt := newAdam(weights)
			workers := newWorkers(cfg, *batch)
			trainStart := time.Now()
			for step := 0; step < *steps; step++ {
				starts := make([]int, *batch)
				for i := range starts {
					starts[i] = tr0 + rng.Intn(trCount)*stride
				}
				loss, acc, gnorm := trainStep(weights, cfg, markov, pre, data, starts, true, workers)
				rate := learningRate(step, 10, *lr)
				clipGrads(workers.grad, 1)
				opt.step(weights, workers.grad, float32(rate))
				lastLoss = loss
				if step%10 == 0 || step+1 == *steps {
					fmt.Printf("step %3d/%d loss=%.4f acc=%.4f grad=%.3f lr=%.2e %s\n",
						step+1, *steps, loss, acc, gnorm, rate, time.Since(trainStart).Round(time.Millisecond))
				}
			}
		}
		if err = saveMarkov(mFile, markov); err != nil {
			panic(err)
		}
		if err = saveWeights(wFile, cfg, weights); err != nil {
			panic(err)
		}
		fmt.Printf("saved %s and %s\n", mFile, wFile)
	}

	after := evaluate(weights, cfg, markov, pre, data, testStarts, false)
	same := markovOnStarts(markov, data, testStarts, cfg.Context, false)
	fmt.Printf("transformer test after  n=%d acc=%.4f ce=%.4f bits=%.4f\n", after.n, after.acc(), after.ce(), after.bits())
	fmt.Printf("markov same windows     n=%d acc=%.4f ce=%.4f bits=%.4f\n", same.n, same.acc(), same.ce(), same.bits())
	if loaded {
		fmt.Printf("held-out ce %.4f\n", after.ce())
	} else {
		fmt.Printf("train last-batch loss=%.4f  held-out ce %.4f -> %.4f (delta %.4f nats)\n",
			lastLoss, before.ce(), after.ce(), before.ce()-after.ce())
	}
	printSample(weights, cfg, markov, pre, data, testStarts[0])

	if *prompt != "" {
		fmt.Printf("sample prompt=%s gen=%d sims=%d temp=%g\n", strconv.Quote(*prompt), *genN, *sims, *temp)
		tGen := time.Now()
		out := newGenerator(cfg, weights, markov, *temp).Search([]byte(*prompt), *genN, *sims, rng)
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
	if after.acc() < 0.3 || after.ce() > 4 || (!loaded && !(after.ce() < before.ce()-0.5)) {
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
			ws:      newWorkspace(cfg, cfg.seqLen()),
			x:       make([]float32, cfg.seqLen()*cfg.Vocab),
			targets: make([]byte, cfg.seqLen()),
		}
	}
	return c
}

func trainStep(w *tensors, cfg Config, m *Markov, pre *distPrefix, data []byte, starts []int, loo bool, c *crew) (loss, acc, gnorm float64) {
	errc := make(chan error, len(starts))
	for i := range starts {
		i := i
		go func() {
			defer func() {
				if r := recover(); r != nil {
					errc <- fmt.Errorf("%v\n%s", r, debug.Stack())
				}
			}()
			s := &c.slots[i]
			m.WriteWindow(data, starts[i], cfg.Context, cfg.Compressed, cfg.Group, loo, pre, s.x, s.targets[cfg.Compressed:])
			errc <- nil
		}()
	}
	for range starts {
		if err := <-errc; err != nil {
			panic(err)
		}
	}
	return forwardSlots(w, cfg, c, len(starts))
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

func evaluate(w *tensors, cfg Config, m *Markov, pre *distPrefix, data []byte, starts []int, loo bool) score {
	var total score
	ws := newWorkspace(cfg, cfg.seqLen())
	x := make([]float32, cfg.seqLen()*cfg.Vocab)
	targets := make([]byte, cfg.seqLen())
	for _, s := range starts {
		m.WriteWindow(data, s, cfg.Context, cfg.Compressed, cfg.Group, loo, pre, x, targets[cfg.Compressed:])
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

func printSample(w *tensors, cfg Config, m *Markov, pre *distPrefix, data []byte, start int) {
	ws := newWorkspace(cfg, cfg.seqLen())
	x := make([]float32, cfg.seqLen()*cfg.Vocab)
	targets := make([]byte, cfg.seqLen())
	m.WriteWindow(data, start, cfg.Context, cfg.Compressed, cfg.Group, false, pre, x, targets[cfg.Compressed:])
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
		row := cfg.Compressed + j
		pred, pp := argmaxRow(ws.logits[row*cfg.Vocab : (row+1)*cfg.Vocab])
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
