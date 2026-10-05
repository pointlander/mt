// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

func TestMul(t *testing.T) {
	A := []float32{1, 2, 3, 4}
	B := []float32{5, 6, 7, 8}
	C := make([]float32, 4)
	mul(A, 2, 2, B, 2, C)
	want := []float32{19, 22, 43, 50}
	for i := range want {
		if C[i] != want[i] {
			t.Fatalf("mul got %v", C)
		}
	}

	// A is [K=2, M=3], B is [K=2, N=2].
	At := []float32{1, 2, 3, 4, 5, 6}
	Bt := []float32{7, 8, 9, 10}
	Ct := make([]float32, 6)
	mulAT(At, 2, 3, Bt, 2, Ct)
	wantT := []float32{43, 48, 59, 66, 75, 84}
	for i := range wantT {
		if Ct[i] != wantT[i] {
			t.Fatalf("mulAT got %v", Ct)
		}
	}

	Ab := []float32{1, 2, 3, 4, 5, 6}
	Bb := []float32{7, 8, 9, 10, 11, 12}
	Cb := make([]float32, 4)
	mulBT(Ab, 2, 3, Bb, 2, Cb)
	wantB := []float32{50, 68, 122, 167}
	for i := range wantB {
		if Cb[i] != wantB[i] {
			t.Fatalf("mulBT got %v", Cb)
		}
	}
}

func TestLayerNormGrad(t *testing.T) {
	const rows, cols = 2, 4
	rng := rand.New(rand.NewSource(2))
	x := make([]float32, rows*cols)
	w := make([]float32, cols)
	b := make([]float32, cols)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	for i := range w {
		w[i] = 1 + float32(rng.NormFloat64())*0.1
		b[i] = float32(rng.NormFloat64()) * 0.1
	}
	y := make([]float32, len(x))
	mean := make([]float32, rows)
	rstd := make([]float32, rows)
	layernorm(x, rows, cols, w, b, y, mean, rstd)
	dy := make([]float32, len(x))
	for i := range dy {
		dy[i] = float32(rng.NormFloat64())
	}
	dx := make([]float32, len(x))
	dw := make([]float32, cols)
	db := make([]float32, cols)
	layernormBackward(dy, x, w, mean, rstd, rows, cols, dx, dw, db)

	loss := func() float32 {
		layernorm(x, rows, cols, w, b, y, mean, rstd)
		var s float32
		for i := range y {
			s += y[i] * dy[i]
		}
		return s
	}
	const eps = 1e-3
	check := func(val *float32, analytic float32, name string) {
		orig := *val
		*val = orig + eps
		lp := loss()
		*val = orig - eps
		lm := loss()
		*val = orig
		numeric := (lp - lm) / (2 * eps)
		if math.Abs(float64(numeric-analytic)) > 2e-2 {
			t.Errorf("%s numeric %g analytic %g", name, numeric, analytic)
		}
	}
	for i := range x {
		check(&x[i], dx[i], "x")
	}
	for i := range w {
		check(&w[i], dw[i], "w")
		check(&b[i], db[i], "b")
	}
}

func TestMarkovBasic(t *testing.T) {
	m := newMarkov()
	text := []byte("0123456789")
	var data []byte
	for i := 0; i < 50; i++ {
		data = append(data, text...)
	}
	m.Train(data, len(data))
	var ctx markovKey
	copy(ctx[:], []byte("0123"))
	p, hit := m.Score(ctx, '4', false)
	// 50 observations of this context, additive smoothing over 256 bins.
	want := float32(50+alpha) / float32(50+alpha*vocab)
	if !hit || math.Abs(float64(p-want)) > 1e-5 {
		t.Fatalf("P('4'|0123)=%g hit=%v, want %g", p, hit, want)
	}
	dist := make([]float32, vocab)
	m.Dist(ctx, '4', false, dist)
	var sum float32
	mode := 0
	for i, v := range dist {
		sum += v
		if v > dist[mode] {
			mode = i
		}
	}
	if math.Abs(float64(sum-1)) > 1e-4 {
		t.Fatalf("dist sum %g", sum)
	}
	if math.Abs(float64(dist['4']-p)) > 1e-5 || mode != '4' || !hit {
		t.Fatalf("score and dist disagree p=%g dist=%g mode=%d hit=%v", p, dist['4'], mode, hit)
	}

	once := newMarkov()
	once.Train([]byte("0123456789abcdef"), 16)
	copy(ctx[:], []byte("0123"))
	loo := make([]float32, vocab)
	once.Dist(ctx, '4', true, loo)
	for i, v := range loo {
		if math.Abs(float64(v-1.0/vocab)) > 1e-5 {
			t.Fatalf("loo bin %d = %g, want uniform", i, v)
		}
	}
}

func TestMarkovBackoff(t *testing.T) {
	m := newMarkov()
	var data []byte
	for i := 0; i < 20; i++ {
		data = append(data, []byte("hello")...)
	}
	m.Train(data, len(data))
	n4, n3, n2, n1 := m.Orders()
	if n4 == 0 || n3 == 0 || n2 == 0 || n1 == 0 {
		t.Fatalf("missing orders 4=%d 3=%d 2=%d 1=%d", n4, n3, n2, n1)
	}

	// "xell" was never counted, but the suffix "ell" predicts 'o'.
	var ctx markovKey
	copy(ctx[:], []byte("xell"))
	p, hit := m.Score(ctx, 'o', false)
	want := float32(20+alpha) / float32(20+alpha*vocab)
	if !hit || math.Abs(float64(p-want)) > 1e-5 {
		t.Fatalf("backoff to order 3: p=%g hit=%v want %g", p, hit, want)
	}

	// Orders 4 and 3 miss. "ab" always continues with 'X', while "b" also
	// continues with 'Y', so a correct backoff stops at width 2.
	m2 := newMarkov()
	var mixed []byte
	for i := 0; i < 10; i++ {
		mixed = append(mixed, []byte("abXcbY")...)
	}
	m2.Train(mixed, len(mixed))
	copy(ctx[:], []byte("zzab"))
	p, hit = m2.Score(ctx, 'X', false)
	want = float32(10+alpha) / float32(10+alpha*vocab)
	if !hit || math.Abs(float64(p-want)) > 1e-5 {
		t.Fatalf("backoff to order 2: p=%g hit=%v want %g", p, hit, want)
	}

	// Order 4 is a singleton, so leave-one-out falls through to the shared suffix.
	back := newMarkov()
	text := []byte("WaaaXZaaaX")
	back.Train(text, len(text))
	copy(ctx[:], []byte("Waaa"))
	p, hit = back.Score(ctx, 'X', true)
	// "aaa" -> 'X' twice. Dropping this event leaves one.
	want = float32(1+alpha) / float32(1+alpha*vocab)
	if !hit || math.Abs(float64(p-want)) > 1e-5 {
		t.Fatalf("loo backoff p=%g hit=%v want %g", p, hit, want)
	}

	// A context that was never seen at any width is uniform.
	copy(ctx[:], []byte("zzzz"))
	uniform := make([]float32, vocab)
	back.Dist(ctx, 'q', false, uniform)
	for i, v := range uniform {
		if math.Abs(float64(v-1.0/vocab)) > 1e-5 {
			t.Fatalf("unseen context bin %d = %g", i, v)
		}
	}
}

func TestWindows(t *testing.T) {
	s0, n := countWindows(100, 0, 80, 3, 4)
	if s0 != 0 || n != 17 {
		t.Fatalf("train windows %d %d", s0, n)
	}
	// s = 0,4,...,64. s=64 targets 68,72,76. s=68 targets 72,76,80 which leaves the region.
	if last := s0 + (n-1)*4; last != 64 {
		t.Fatalf("last train start %d", last)
	}
	s0, n = countWindows(100, 80, 100, 3, 4)
	if s0 != 76 || n != 3 {
		t.Fatalf("test windows %d %d", s0, n)
	}
}

func TestCausal(t *testing.T) {
	cfg := Config{Vocab: 8, Context: 4, D: 8, Heads: 2, DFF: 16}
	rng := rand.New(rand.NewSource(3))
	w := newTensors(cfg)
	initTensors(w, cfg, rng)
	ws := newWorkspace(cfg, cfg.Context)
	x := simplex(rng, cfg.Context, cfg.Vocab)
	targets := []byte{1, 2, 3, 4}
	forwardBackward(cfg, w, nil, ws, x, targets)
	first := append([]float32(nil), ws.logits[:cfg.Vocab]...)
	x[3*cfg.Vocab] += 0.5
	forwardBackward(cfg, w, nil, ws, x, targets)
	for i, v := range first {
		if v != ws.logits[i] {
			t.Fatalf("future input changed position 0 logit %d: %g vs %g", i, v, ws.logits[i])
		}
	}
}

func TestGradients(t *testing.T) {
	cfg := Config{Vocab: 6, Context: 3, D: 8, Heads: 2, DFF: 8}
	rng := rand.New(rand.NewSource(4))
	w := newTensors(cfg)
	initTensors(w, cfg, rng)
	g := newTensors(cfg)
	ws := newWorkspace(cfg, cfg.Context)
	x := simplex(rng, cfg.Context, cfg.Vocab)
	targets := []byte{1, 2, 0}
	forwardBackward(cfg, w, g, ws, x, targets)
	analytic := snapshot(g)

	const eps = 1.0 / 512
	names := []string{
		"inW", "inB", "pos", "ln1W", "ln1B", "wq", "wk", "wv", "wo",
		"bq", "bk", "bv", "bo", "ln2W", "ln2B", "w1", "b1", "w2", "b2",
		"lnFW", "lnFB", "wh", "bh",
	}
	wsList := w.list()
	failures := 0
	for pi, p := range wsList {
		ag := analytic[pi]
		for i := range p {
			orig := p[i]
			p[i] = orig + eps
			lp, _ := forwardBackward(cfg, w, nil, ws, x, targets)
			p[i] = orig - eps
			lm, _ := forwardBackward(cfg, w, nil, ws, x, targets)
			p[i] = orig
			numeric := float32((lp - lm) / (2 * eps))
			diff := math.Abs(float64(numeric - ag[i]))
			scale := math.Max(math.Abs(float64(numeric)), math.Abs(float64(ag[i])))
			if diff > 2e-2 && diff > 0.05*math.Max(scale, 1e-3) {
				failures++
				if failures <= 12 {
					t.Errorf("%s[%d] numeric %g analytic %g", names[pi], i, numeric, ag[i])
				}
			}
		}
	}
	if failures > 0 {
		t.Fatalf("%d gradient mismatches", failures)
	}
}

func TestOverfit(t *testing.T) {
	pattern := []byte("abcdefghijklmnopqrstuvwxyz")
	var data []byte
	for i := 0; i < 80; i++ {
		data = append(data, pattern...)
	}
	trainEnd := int(float64(len(data)) * 0.8)
	m := newMarkov()
	m.Train(data, trainEnd)
	cfg := Config{Vocab: vocab, Context: 16, D: 32, Heads: 4, DFF: 64}
	rng := rand.New(rand.NewSource(5))
	w := newTensors(cfg)
	initTensors(w, cfg, rng)
	s0, count := countWindows(len(data), 0, trainEnd, cfg.Context, stride)
	if count < 2 {
		t.Fatal("no windows")
	}
	opt := newAdam(w)
	ws := newWorkspace(cfg, cfg.Context)
	g := newTensors(cfg)
	slot := newTensors(cfg)
	x := make([]float32, cfg.Context*cfg.Vocab)
	targets := make([]byte, cfg.Context)
	m.Fill(data, s0, cfg.Context, true, x, targets)
	first, _ := forwardBackward(cfg, w, nil, ws, x, targets)
	const batch = 4
	for step := 0; step < 40; step++ {
		g.zero()
		for b := 0; b < batch; b++ {
			s := s0 + rng.Intn(count)*stride
			m.Fill(data, s, cfg.Context, true, x, targets)
			forwardBackward(cfg, w, slot, ws, x, targets)
			addTensors(g, slot)
		}
		scaleTensors(g, 1/float32(batch))
		clipGrads(g, 1)
		opt.step(w, g, 2e-3)
	}
	m.Fill(data, s0, cfg.Context, true, x, targets)
	last, correct := forwardBackward(cfg, w, nil, ws, x, targets)
	if !(last < first-0.5) {
		t.Fatalf("loss did not fall: %g -> %g", first, last)
	}
	if correct < cfg.Context/2 {
		t.Fatalf("accuracy %d/%d after overfit", correct, cfg.Context)
	}
}

func TestMarkovPG100(t *testing.T) {
	data, err := os.ReadFile("pg100.txt")
	if err != nil {
		t.Fatal(err)
	}
	trainEnd := int(float64(len(data)) * trainFraction)
	m := newMarkov()
	m.Train(data, trainEnd)
	test := m.Evaluate(data, trainEnd, len(data), 1, false)
	if test.acc() < 0.2 || test.ce() > 4 {
		t.Fatalf("markov test acc=%g ce=%g", test.acc(), test.ce())
	}
	strideScore := m.Evaluate(data, trainEnd, len(data), stride, false)
	if strideScore.n < 1000 || strideScore.acc() < 0.2 {
		t.Fatalf("stride score %+v acc=%g", strideScore, strideScore.acc())
	}
}

func simplex(rng *rand.Rand, rows, cols int) []float32 {
	x := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		var s float32
		row := x[i*cols : (i+1)*cols]
		for j := range row {
			row[j] = float32(rng.Float64()) + 0.01
			s += row[j]
		}
		for j := range row {
			row[j] /= s
		}
	}
	return x
}

func snapshot(g *tensors) [][]float32 {
	out := make([][]float32, 0)
	for _, p := range g.list() {
		out = append(out, append([]float32(nil), p...))
	}
	return out
}
