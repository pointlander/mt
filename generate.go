// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
)

// A continuation is drawn one byte at a time from the tempered softmax.
// Temperature 1 leaves the logits unchanged. The sample line is one draw.
// Each MCTS simulation is another independent draw from that same softmax,
// and the reported MCTS continuation is the last simulation. With sims == 0
// the MCTS line repeats the sample draw. The inputs to the softmax are the
// order-4 Markov rows aligned to the byte being predicted.

type continuation struct {
	sample     []byte
	sampleLogp float64
	text       []byte
	logp       float64
}

// searchContinuation draws continuations from next. logp is the log
// probability of the drawn bytes under next.
func searchContinuation(prompt []byte, length, sims int, rng *rand.Rand, next func(prefix []byte) []float32) continuation {
	if length < 1 {
		panic("generation length")
	}
	if sims < 0 {
		panic("sims")
	}
	sampled, sampleLogp := drawContinuation(prompt, length, rng, next)
	text, logp := sampled, sampleLogp
	for i := 0; i < sims; i++ {
		text, logp = drawContinuation(prompt, length, rng, next)
	}
	return continuation{
		sample:     sampled,
		sampleLogp: sampleLogp,
		text:       append([]byte(nil), text...),
		logp:       logp,
	}
}

func drawContinuation(prompt []byte, length int, rng *rand.Rand, next func(prefix []byte) []float32) ([]byte, float64) {
	cur := append([]byte(nil), prompt...)
	text := make([]byte, 0, length)
	var logp float64
	for i := 0; i < length; i++ {
		b, p := sampleDist(next(cur), rng)
		text = append(text, b)
		logp += math.Log(p)
		cur = append(cur, b)
	}
	return text, logp
}

// sampleDist draws a byte with probability proportional to dist.
func sampleDist(dist []float32, rng *rand.Rand) (byte, float64) {
	var sum float64
	for _, v := range dist {
		if v > 0 {
			sum += float64(v)
		}
	}
	if sum == 0 {
		panic("empty next-byte distribution")
	}
	r := rng.Float64() * sum
	var acc float64
	last, lastP := byte(0), 0.0
	for i, v := range dist {
		if v <= 0 {
			continue
		}
		last = byte(i)
		lastP = float64(v)
		acc += lastP
		if r <= acc {
			return last, lastP
		}
	}
	return last, lastP
}

// generator caches transformer next-byte distributions for MCTS.
// temperature scales the logits before the softmax. It is fixed for the
// generator, so the cache stores the tempered distribution.
type generator struct {
	cfg         Config
	w           *tensors
	m           *Markov
	ws          *workspace
	x           []float32
	targets     []byte
	cache       map[string][]float32
	temperature float64
}

func newGenerator(cfg Config, w *tensors, m *Markov, temperature float64) *generator {
	if !(temperature > 0) {
		panic("temperature")
	}
	return &generator{
		cfg:         cfg,
		w:           w,
		m:           m,
		ws:          newWorkspace(cfg, cfg.seqLen()),
		x:           make([]float32, cfg.seqLen()*cfg.Vocab),
		targets:     make([]byte, cfg.seqLen()),
		cache:       make(map[string][]float32),
		temperature: temperature,
	}
}

// Next is the tempered transformer distribution for the byte after prefix.
// Compressed sums of earlier rows are prepended when the model has them.
// The cache key is the suffix of prefix that those rows actually read.
func (g *generator) Next(prefix []byte) []float32 {
	if len(prefix) < order {
		panic("prompt shorter than 4 bytes")
	}
	nvec := len(prefix) / order
	raw := nvec
	if raw > g.cfg.Context {
		raw = g.cfg.Context
	}
	compressed := g.cfg.Compressed
	hist := 0
	if compressed > 0 {
		hist = compressed * g.cfg.Group
		if hist > nvec-raw {
			hist = nvec - raw
		}
	}
	used := (hist + raw) * order
	key := string(prefix[len(prefix)-used:])
	if d, ok := g.cache[key]; ok {
		return d
	}
	T := compressed + raw
	s := len(prefix) - raw*order
	g.m.WriteWindow(prefix, s, raw, compressed, g.cfg.Group, false, nil, g.x[:T*vocab], g.targets[compressed:T])
	forwardBackward(g.cfg, g.w, nil, g.ws, g.x[:T*vocab], g.targets[:T])
	probs := softmax(g.ws.logits[(T-1)*vocab:T*vocab], g.temperature)
	g.cache[key] = probs
	return probs
}

func (g *generator) Search(prompt []byte, length, sims int, rng *rand.Rand) continuation {
	return searchContinuation(prompt, length, sims, rng, g.Next)
}

// softmax is the next-byte distribution at a positive temperature.
// Logits are divided by temperature. 1 leaves them unchanged, values above
// 1 flatten the distribution, and values below 1 sharpen it.
func softmax(logits []float32, temperature float64) []float32 {
	if !(temperature > 0) {
		panic("temperature")
	}
	invT := 1 / temperature
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	out := make([]float32, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v-max) * invT)
		out[i] = float32(e)
		sum += e
	}
	inv := float32(1 / sum)
	for i := range out {
		out[i] *= inv
	}
	return out
}
