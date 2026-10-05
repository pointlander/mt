// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"testing"
)

// trap is a next-byte model whose greedy decode is not the maximum-probability
// string. From the empty prefix, 'a' is more likely than 'b'. After 'a' every
// byte is uniform. After 'b' the next byte is 'y' with probability 1.
func trap(prefix []byte) []float32 {
	p := make([]float32, vocab)
	switch {
	case len(prefix) == 0:
		p['a'] = 0.6
		p['b'] = 0.4
	case prefix[len(prefix)-1] == 'b':
		p['y'] = 1
	default:
		u := float32(1) / vocab
		for i := range p {
			p[i] = u
		}
	}
	return p
}

func TestMCTSBeatsGreedy(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	got := searchContinuation(nil, 2, 8, 2, rng, trap)
	if string(got.greedy) != "a\x00" {
		t.Fatalf("greedy %q", got.greedy)
	}
	if string(got.text) != "by" {
		t.Fatalf("mcts %q logp=%g greedyLogp=%g", got.text, got.logp, got.greedyLogp)
	}
	// Priors are stored as float32, so the log uses that rounding.
	want := math.Log(float64(float32(0.4))) + math.Log(1)
	if math.Abs(got.logp-want) > 1e-9 {
		t.Fatalf("logp %g want %g", got.logp, want)
	}
	if !(got.logp > got.greedyLogp) {
		t.Fatalf("mcts did not beat greedy: %g vs %g", got.logp, got.greedyLogp)
	}
}

func TestContinuationLogp(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	prompt := []byte("hi")
	next := func(prefix []byte) []float32 {
		p := make([]float32, vocab)
		p['x'] = 0.7
		p['z'] = 0.3
		return p
	}
	got := searchContinuation(prompt, 3, 0, 2, rng, next)
	if string(got.text) != "xxx" || string(got.greedy) != "xxx" {
		t.Fatalf("got %q greedy %q", got.text, got.greedy)
	}
	want := 3 * math.Log(float64(float32(0.7)))
	if math.Abs(got.logp-want) > 1e-9 {
		t.Fatalf("logp %g want %g", got.logp, want)
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
	g := newGenerator(cfg, w, m)
	prompt := []byte("abcd")
	got := g.Search(prompt, 4, 4, 4, rng)
	if len(got.text) != 4 || len(got.greedy) != 4 {
		t.Fatalf("lengths %d %d", len(got.text), len(got.greedy))
	}
	if math.IsNaN(got.logp) || math.IsInf(got.logp, 0) {
		t.Fatalf("logp %g", got.logp)
	}
	if got.logp+1e-6 < got.greedyLogp {
		t.Fatalf("search %g lost to greedy %g", got.logp, got.greedyLogp)
	}
	again := continuationLogp(prompt, got.text, g.Next)
	if math.Abs(again-got.logp) > 1e-6 {
		t.Fatalf("recomputed %g != %g", again, got.logp)
	}
}

func TestDiversifyBreaksRuns(t *testing.T) {
	raw := make([]float32, vocab)
	raw['a'] = 1

	// Three identical bytes already: the fourth copy is removed, and a
	// delta on that byte must not renorm back to 1.
	got := diversify(raw, []byte("aaa"))
	if got['a'] != 0 {
		t.Fatalf("run of three left 'a' at %g", got['a'])
	}
	u := float32(1) / float32(vocab-1)
	var sum float64
	for i, v := range got {
		sum += float64(v)
		if i == int('a') {
			continue
		}
		if v != u {
			t.Fatalf("byte %d = %g want %g", i, v, u)
		}
	}
	if math.Abs(sum-1) > 1e-5 {
		t.Fatalf("sum %g", sum)
	}

	// Two copies are still free, so "www" and a double letter can finish.
	still := diversify(raw, []byte("aa"))
	if still['a'] < 0.99 {
		t.Fatalf("run of two was blocked: %g", still['a'])
	}
}

func TestDiversifyPeriod(t *testing.T) {
	raw := make([]float32, vocab)
	raw['r'] = 0.5
	raw['z'] = 0.5

	// "rere" is the pattern twice: discourage another 'r', but keep it.
	soft := diversify(raw, []byte("rere"))
	if soft['r'] == 0 || !(soft['r'] < soft['z']) {
		t.Fatalf("soft r=%g z=%g", soft['r'], soft['z'])
	}
	var sum float64
	for _, v := range soft {
		sum += float64(v)
	}
	if math.Abs(sum-1) > 1e-5 {
		t.Fatalf("soft sum %g", sum)
	}

	// "rerere" is the pattern three times: the next copy is impossible.
	hard := diversify(raw, []byte("rerere"))
	if hard['r'] != 0 {
		t.Fatalf("rerere left r=%g", hard['r'])
	}
	if hard['z'] <= hard['r'] {
		t.Fatalf("hard z=%g r=%g", hard['z'], hard['r'])
	}

	// "banan" has one "an" repeat. A confident model may still finish "banana".
	word := make([]float32, vocab)
	word['a'] = 0.9
	word['z'] = 0.1
	done := diversify(word, []byte("banan"))
	if done['a'] == 0 || !(done['a'] > done['z']) {
		t.Fatalf("banana completion a=%g z=%g", done['a'], done['z'])
	}
}

func TestSearchBreaksRepeat(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	raw := make([]float32, vocab)
	raw['z'] = 1
	next := func(prefix []byte) []float32 {
		return diversify(raw, prefix)
	}
	got := searchContinuation([]byte("zzz"), 8, 4, 4, rng, next)
	if string(got.greedy) == "zzzzzzzz" || string(got.text) == "zzzzzzzz" {
		t.Fatalf("run survived greedy %q mcts %q", got.greedy, got.text)
	}
	if maxRun(append([]byte("zzz"), got.greedy...)) >= 4 || maxRun(append([]byte("zzz"), got.text...)) >= 4 {
		t.Fatalf("run of 4 greedy %q mcts %q", got.greedy, got.text)
	}
}

func maxRun(s []byte) int {
	best, cur := 1, 1
	for i := 1; i < len(s); i++ {
		if s[i] == s[i-1] {
			cur++
			if cur > best {
				best = cur
			}
			continue
		}
		cur = 1
	}
	return best
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
