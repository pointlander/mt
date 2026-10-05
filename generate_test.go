// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"testing"
)

func xz(prefix []byte) []float32 {
	p := make([]float32, vocab)
	p['x'] = 0.7
	p['z'] = 0.3
	return p
}

func TestDeltaSample(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	next := func(prefix []byte) []float32 {
		p := make([]float32, vocab)
		p['x'] = 1
		return p
	}
	got := searchContinuation([]byte("hi"), 3, 4, rng, next)
	if string(got.sample) != "xxx" || string(got.text) != "xxx" {
		t.Fatalf("sample %q mcts %q", got.sample, got.text)
	}
	if got.sampleLogp != 0 || got.logp != 0 {
		t.Fatalf("logp sample %g mcts %g", got.sampleLogp, got.logp)
	}
}

func TestZeroSimsSharesSample(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	prompt := []byte("hi")
	got := searchContinuation(prompt, 5, 0, rng, xz)
	if string(got.sample) != string(got.text) || got.sampleLogp != got.logp {
		t.Fatalf("sample %q (%g) mcts %q (%g)", got.sample, got.sampleLogp, got.text, got.logp)
	}
	if math.Abs(continuationLogp(prompt, got.text, xz)-got.logp) > 1e-9 {
		t.Fatalf("logp %g", got.logp)
	}
}

func TestSamplesFromSoftmax(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	prompt := []byte("hi")
	var nx, n int
	for i := 0; i < 20; i++ {
		got := searchContinuation(prompt, 8, 1, rng, xz)
		for _, part := range [][]byte{got.sample, got.text} {
			if len(part) != 8 {
				t.Fatalf("length %d", len(part))
			}
			for _, b := range part {
				n++
				switch b {
				case 'x':
					nx++
				case 'z':
				default:
					t.Fatalf("unexpected byte %q", []byte{b})
				}
			}
		}
		if math.Abs(continuationLogp(prompt, got.sample, xz)-got.sampleLogp) > 1e-9 {
			t.Fatalf("sample logp %g", got.sampleLogp)
		}
		if math.Abs(continuationLogp(prompt, got.text, xz)-got.logp) > 1e-9 {
			t.Fatalf("mcts logp %g", got.logp)
		}
	}
	frac := float64(nx) / float64(n)
	if frac > 0.9 || frac < 0.5 {
		t.Fatalf("fraction of x = %g (n=%d), sampling collapsed toward argmax or away from it", frac, n)
	}
}

func TestShortWindowMatches(t *testing.T) {
	cfg := Config{Vocab: 8, Context: 6, D: 8, Heads: 2, DFF: 16}
	rng := rand.New(rand.NewSource(3))
	w := newTensors(cfg)
	initTensors(w, cfg, rng)
	x := simplex(rand.New(rand.NewSource(9)), 3, cfg.Vocab)
	targets := []byte{1, 2, 4}
	big := newWorkspace(cfg, cfg.Context)
	small := newWorkspace(cfg, 3)
	forwardBackward(cfg, w, nil, big, x, targets)
	want := append([]float32(nil), big.logits[:3*cfg.Vocab]...)
	forwardBackward(cfg, w, nil, small, x, targets)
	for i := range want {
		if want[i] != small.logits[i] {
			t.Fatalf("logit %d %g vs %g", i, want[i], small.logits[i])
		}
	}
}

func TestGenerateOnModel(t *testing.T) {
	pattern := []byte("abcdefghijklmnopqrstuvwxyz")
	var data []byte
	for i := 0; i < 40; i++ {
		data = append(data, pattern...)
	}
	m := newMarkov()
	m.Train(data, len(data))
	cfg := Config{Vocab: vocab, Context: 16, D: 32, Heads: 4, DFF: 64}
	rng := rand.New(rand.NewSource(4))
	w := newTensors(cfg)
	initTensors(w, cfg, rng)
	g := newGenerator(cfg, w, m, 1)
	prompt := []byte("abcd")
	got := g.Search(prompt, 4, 4, rng)
	if len(got.text) != 4 || len(got.sample) != 4 {
		t.Fatalf("lengths %d %d", len(got.text), len(got.sample))
	}
	if math.IsNaN(got.logp) || math.IsInf(got.logp, 0) || math.IsNaN(got.sampleLogp) || math.IsInf(got.sampleLogp, 0) {
		t.Fatalf("logp sample %g mcts %g", got.sampleLogp, got.logp)
	}
	if math.Abs(continuationLogp(prompt, got.text, g.Next)-got.logp) > 1e-6 {
		t.Fatalf("recomputed mcts %g != %g", continuationLogp(prompt, got.text, g.Next), got.logp)
	}
	if math.Abs(continuationLogp(prompt, got.sample, g.Next)-got.sampleLogp) > 1e-6 {
		t.Fatalf("recomputed sample %g != %g", continuationLogp(prompt, got.sample, g.Next), got.sampleLogp)
	}
}

func TestSoftmaxTemperature(t *testing.T) {
	logits := []float32{0, 1, 3}
	unit := softmax(logits, 1)
	var wantSum float64
	wants := make([]float64, len(logits))
	for i, v := range logits {
		wants[i] = math.Exp(float64(v - logits[2]))
		wantSum += wants[i]
	}
	var sum float64
	for i := range wants {
		wants[i] /= wantSum
		sum += float64(unit[i])
		if math.Abs(float64(unit[i])-wants[i]) > 1e-6 {
			t.Fatalf("T=1 [%d] %g want %g", i, unit[i], wants[i])
		}
	}
	if math.Abs(sum-1) > 1e-5 {
		t.Fatalf("sum %g", sum)
	}
	hot := softmax(logits, 0.5)
	cold := softmax(logits, 2)
	if !(hot[2] > unit[2] && unit[2] > cold[2]) {
		t.Fatalf("peak hot=%g unit=%g cold=%g", hot[2], unit[2], cold[2])
	}
	if !(cold[0] > unit[0] && unit[0] > hot[0]) {
		t.Fatalf("tail cold=%g unit=%g hot=%g", cold[0], unit[0], hot[0])
	}
	for _, dist := range [][]float32{hot, cold} {
		sum = 0
		for _, v := range dist {
			sum += float64(v)
			if v <= 0 {
				t.Fatalf("nonpositive %g", v)
			}
		}
		if math.Abs(sum-1) > 1e-5 {
			t.Fatalf("sum %g", sum)
		}
	}
}

func TestGeneratorTemperatureFlattens(t *testing.T) {
	data := []byte("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz")
	m := newMarkov()
	m.Train(data, len(data))
	cfg := Config{Vocab: vocab, Context: 8, D: 16, Heads: 2, DFF: 32}
	rng := rand.New(rand.NewSource(5))
	w := newTensors(cfg)
	initTensors(w, cfg, rng)
	prefix := []byte("abcd")
	cold := newGenerator(cfg, w, m, 2).Next(prefix)
	hot := newGenerator(cfg, w, m, 0.5).Next(prefix)
	coldMax, hotMax := cold[0], hot[0]
	var coldSum, hotSum float64
	for i := range cold {
		coldSum += float64(cold[i])
		hotSum += float64(hot[i])
		if cold[i] > coldMax {
			coldMax = cold[i]
		}
		if hot[i] > hotMax {
			hotMax = hot[i]
		}
	}
	if !(hotMax > coldMax) {
		t.Fatalf("lower temperature was not sharper: hot %g cold %g", hotMax, coldMax)
	}
	if math.Abs(coldSum-1) > 1e-4 || math.Abs(hotSum-1) > 1e-4 {
		t.Fatalf("sums %g %g", coldSum, hotSum)
	}
}

func continuationLogp(prompt, text []byte, next func([]byte) []float32) float64 {
	cur := append([]byte(nil), prompt...)
	var logp float64
	for _, b := range text {
		p := next(cur)[b]
		logp += math.Log(float64(p))
		cur = append(cur, b)
	}
	return logp
}
