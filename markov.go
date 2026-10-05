// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

const (
	// order is the Markov conditioning width, in bytes.
	order = 4
	// vocab is the byte alphabet.
	vocab = 256
	// alpha is the additive count mixed into every next-byte bin.
	// It is small so a repeated 4-gram stays peaked, and large enough
	// that an unseen byte still has positive probability.
	alpha = 0.01
	// stride is the spacing between distributions fed to the transformer.
	// It matches the Markov order so successive inputs read disjoint 4-grams.
	stride = 4
)

type markovKey [order]byte

type markovEntry struct {
	sym   []byte
	cnt   []uint32
	total uint32
}

// Markov is a 4th-order next-byte model. Distributions use additive smoothing.
type Markov struct {
	tab          map[markovKey]*markovEntry
	observations int
}

func newMarkov() *Markov {
	return &Markov{tab: make(map[markovKey]*markovEntry, 1<<16)}
}

// Observe records one transition ctx -> next.
func (m *Markov) Observe(ctx markovKey, next byte) {
	e := m.tab[ctx]
	if e == nil {
		e = &markovEntry{}
		m.tab[ctx] = e
	}
	for i, s := range e.sym {
		if s == next {
			e.cnt[i]++
			e.total++
			m.observations++
			return
		}
	}
	e.sym = append(e.sym, next)
	e.cnt = append(e.cnt, 1)
	e.total++
	m.observations++
}

// Train counts every order-4 transition in data[0:end].
func (m *Markov) Train(data []byte, end int) {
	if end > len(data) {
		end = len(data)
	}
	var ctx markovKey
	for i := order; i < end; i++ {
		copy(ctx[:], data[i-order:i])
		m.Observe(ctx, data[i])
	}
}

func (m *Markov) entryCounts(ctx markovKey, target byte, loo bool) (total uint32, targetCount uint32, best byte, bestCount uint32, seen bool) {
	e := m.tab[ctx]
	if e == nil {
		return 0, 0, 0, 0, false
	}
	total = e.total
	if loo {
		if total == 0 {
			panic("leave-one-out on an empty context")
		}
		total--
	}
	for i, s := range e.sym {
		c := e.cnt[i]
		if loo && s == target {
			if c == 0 {
				panic("leave-one-out missing the target")
			}
			c--
		}
		if s == target {
			targetCount = c
		}
		if c == 0 {
			continue
		}
		if !seen || c > bestCount || (c == bestCount && s < best) {
			best = s
			bestCount = c
			seen = true
		}
	}
	return total, targetCount, best, bestCount, seen
}

// Dist writes P(next | ctx). If loo is set, one count of target is removed
// so the label being predicted is not inside its own input distribution.
func (m *Markov) Dist(ctx markovKey, target byte, loo bool, out []float32) {
	if len(out) != vocab {
		panic("dist length")
	}
	total, _, _, _, _ := m.entryCounts(ctx, target, loo)
	denom := float32(total) + alpha*float32(vocab)
	base := alpha / denom
	for i := range out {
		out[i] = base
	}
	e := m.tab[ctx]
	if e == nil {
		return
	}
	inv := 1 / denom
	for i, s := range e.sym {
		c := e.cnt[i]
		if loo && s == target {
			c--
		}
		out[s] = (float32(c) + alpha) * inv
	}
}

// Score returns P(target | ctx) and whether target is the mode.
// Ties break toward the smaller byte. An all-tied row predicts byte 0.
func (m *Markov) Score(ctx markovKey, target byte, loo bool) (p float32, hit bool) {
	total, targetCount, best, _, seen := m.entryCounts(ctx, target, loo)
	denom := float32(total) + alpha*float32(vocab)
	p = (float32(targetCount) + alpha) / denom
	if !seen {
		best = 0
	}
	return p, best == target
}

type score struct {
	nll     float64
	correct int
	n       int
}

func (s score) add(o score) score {
	s.nll += o.nll
	s.correct += o.correct
	s.n += o.n
	return s
}

func (s score) ce() float64 {
	if s.n == 0 {
		return math.NaN()
	}
	return s.nll / float64(s.n)
}

func (s score) acc() float64 {
	if s.n == 0 {
		return math.NaN()
	}
	return float64(s.correct) / float64(s.n)
}

func (s score) bits() float64 { return s.ce() / math.Ln2 }

// Evaluate scores next-byte predictions for targets in [lo, hi).
// stride 1 visits every byte. stride 4 visits the transformer's targets,
// which sit on multiples of 4.
func (m *Markov) Evaluate(data []byte, lo, hi, step int, loo bool) score {
	if step < 1 {
		panic("step")
	}
	if hi > len(data) {
		hi = len(data)
	}
	i0 := lo
	if i0 < order {
		i0 = order
	}
	if step > 1 {
		if r := i0 % step; r != 0 {
			i0 += step - r
		}
	}
	var out score
	var ctx markovKey
	for i := i0; i < hi; i += step {
		copy(ctx[:], data[i-order:i])
		p, hit := m.Score(ctx, data[i], loo)
		out.n++
		out.nll += -math.Log(float64(p))
		if hit {
			out.correct++
		}
	}
	return out
}

// Fill writes a transformer window opened at byte offset s.
// Position j gets P(· | data[s+4j : s+4j+4]) and target data[s+4j+4].
func (m *Markov) Fill(data []byte, s, T int, loo bool, x []float32, targets []byte) {
	if len(targets) != T || len(x) != T*vocab {
		panic("fill length")
	}
	var ctx markovKey
	for j := 0; j < T; j++ {
		off := s + stride*j
		if off+stride >= len(data) {
			panic("fill past end")
		}
		copy(ctx[:], data[off:off+order])
		tgt := data[off+order]
		targets[j] = tgt
		m.Dist(ctx, tgt, loo, x[j*vocab:(j+1)*vocab])
	}
}
