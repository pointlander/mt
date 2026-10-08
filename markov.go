// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

const (
	// order is the longest Markov conditioning width, in bytes.
	// Contexts of length order-1 down to 1 are stored too and used on a miss.
	order = 4
	// vocab is the byte alphabet.
	vocab = 256
	// alpha is the additive count mixed into every next-byte bin.
	// It is small so a repeated context stays peaked, and large enough
	// that an unseen byte still has positive probability.
	alpha = 0.01
	// stride is the spacing between distributions fed to the transformer.
	// It matches the longest Markov order so successive inputs read disjoint 4-grams.
	stride = 4
)

// markovKey is the order-4 context used by callers. Shorter lookups use its suffix.
type markovKey [order]byte

// ctxKey is a context of length n. b[:n] holds the bytes from oldest to newest.
type ctxKey struct {
	n byte
	b [order]byte
}

type markovEntry struct {
	sym   []byte
	cnt   []uint32
	total uint32
}

func (e *markovEntry) count(sym byte) uint32 {
	for i, s := range e.sym {
		if s == sym {
			return e.cnt[i]
		}
	}
	return 0
}

// Markov is a next-byte model over contexts of 4, 3, 2, and 1 bytes.
// A lookup uses the longest context that was actually counted, then the next
// shorter suffix. Distributions use additive smoothing.
type Markov struct {
	tab          map[ctxKey]*markovEntry
	observations int
}

func newMarkov() *Markov {
	return &Markov{tab: make(map[ctxKey]*markovEntry, 1<<17)}
}

func suffixKey(ctx markovKey, n int) ctxKey {
	var k ctxKey
	k.n = byte(n)
	copy(k.b[:n], ctx[order-n:])
	return k
}

func (e *markovEntry) add(next byte) {
	for i, s := range e.sym {
		if s == next {
			e.cnt[i]++
			e.total++
			return
		}
	}
	e.sym = append(e.sym, next)
	e.cnt = append(e.cnt, 1)
	e.total++
}

func (m *Markov) observe(key ctxKey, next byte) {
	e := m.tab[key]
	if e == nil {
		e = &markovEntry{}
		m.tab[key] = e
	}
	e.add(next)
}

// Train counts next-byte transitions for every context length from 1 through order.
func (m *Markov) Train(data []byte, end int) {
	if end > len(data) {
		end = len(data)
	}
	for i := 1; i < end; i++ {
		nctx := i
		if nctx > order {
			nctx = order
		}
		var ctx markovKey
		copy(ctx[order-nctx:], data[i-nctx:i])
		next := data[i]
		for n := nctx; n >= 1; n-- {
			m.observe(suffixKey(ctx, n), next)
		}
	}
	if end > 1 {
		m.observations += end - 1
	}
}

// Orders returns the number of stored contexts of length 4, 3, 2, and 1.
func (m *Markov) Orders() (n4, n3, n2, n1 int) {
	for k := range m.tab {
		switch k.n {
		case 4:
			n4++
		case 3:
			n3++
		case 2:
			n2++
		case 1:
			n1++
		}
	}
	return n4, n3, n2, n1
}

// resolve picks the context distribution for ctx. A length misses when it was
// never stored. Leave-one-out also misses when removing target drops the last
// count, and the search continues with the next shorter suffix.
func (m *Markov) resolve(ctx markovKey, target byte, loo bool) (e *markovEntry, drop bool) {
	for n := order; n >= 1; n-- {
		e = m.tab[suffixKey(ctx, n)]
		if e == nil {
			continue
		}
		if !loo {
			return e, false
		}
		c := e.count(target)
		if c == 0 {
			return e, false
		}
		if e.total <= 1 {
			continue
		}
		return e, true
	}
	return nil, false
}

// Dist writes P(next | ctx). If loo is set, one count of target is removed
// from the context that supplies the distribution, so the label being
// predicted is not inside its own input distribution.
func (m *Markov) Dist(ctx markovKey, target byte, loo bool, out []float32) {
	if len(out) != vocab {
		panic("dist length")
	}
	e, drop := m.resolve(ctx, target, loo)
	var total uint32
	if e != nil {
		total = e.total
		if drop {
			total--
		}
	}
	denom := float32(total) + alpha*float32(vocab)
	base := alpha / denom
	for i := range out {
		out[i] = base
	}
	if e == nil {
		return
	}
	inv := 1 / denom
	for i, s := range e.sym {
		c := e.cnt[i]
		if drop && s == target {
			c--
		}
		out[s] = (float32(c) + alpha) * inv
	}
}

// Score returns P(target | ctx) and whether target is the mode.
// Ties break toward the smaller byte. An all-tied row predicts byte 0.
func (m *Markov) Score(ctx markovKey, target byte, loo bool) (p float32, hit bool) {
	e, drop := m.resolve(ctx, target, loo)
	var total uint32
	var targetCount uint32
	best := byte(0)
	var bestCount uint32
	seen := false
	if e != nil {
		total = e.total
		if drop {
			total--
		}
		for i, s := range e.sym {
			c := e.cnt[i]
			if drop && s == target {
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
	}
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

// PredictWindow writes T Markov distributions aligned to predict the byte at
// index n. data[:n] is the prefix; the target byte itself is not read.
// The last row is P(· | data[n-4:n]).
func (m *Markov) PredictWindow(data []byte, n, T int, x []float32) {
	if T < 1 || n < order*T || len(data) < n || len(x) < T*vocab {
		panic("predict window")
	}
	start := n - order*T
	var ctx markovKey
	for j := 0; j < T; j++ {
		off := start + order*j
		copy(ctx[:], data[off:off+order])
		m.Dist(ctx, 0, false, x[j*vocab:(j+1)*vocab])
	}
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

// distPrefix holds exclusive prefix sums of the stride-aligned raw rows.
// sum[(k+1)*vocab:(k+2)*vocab] is the sum of rows [0, k].
// Rows are non-leave-one-out. A compressed bucket is a difference of two entries.
type distPrefix struct {
	sum []float64
	n   int
}

func (m *Markov) buildDistPrefix(data []byte) *distPrefix {
	n := 0
	if len(data) > order {
		n = (len(data) - order) / stride
	}
	p := &distPrefix{sum: make([]float64, (n+1)*vocab), n: n}
	var ctx markovKey
	var row [vocab]float32
	for k := 0; k < n; k++ {
		off := k * stride
		copy(ctx[:], data[off:off+order])
		m.Dist(ctx, 0, false, row[:])
		src := p.sum[k*vocab:]
		dst := p.sum[(k+1)*vocab:]
		for b, v := range row {
			dst[b] = src[b] + float64(v)
		}
	}
	return p
}

// sumInto writes the sum of raw rows [lo, hi) into out.
// Indexes before 0 or at or after n contribute nothing.
func (p *distPrefix) sumInto(lo, hi int, out []float32) {
	if len(out) != vocab {
		panic("sum length")
	}
	if lo < 0 {
		lo = 0
	}
	if hi > p.n {
		hi = p.n
	}
	if hi <= lo {
		clear(out)
		return
	}
	a := p.sum[lo*vocab:]
	b := p.sum[hi*vocab:]
	for i := range out {
		out[i] = float32(b[i] - a[i])
	}
}

// WriteWindow writes compressed sums and then raw rows for a window opened at s.
// Bucket i, oldest first, is the sum of Group raw rows in the block of
// Compressed*Group rows immediately before s. Rows that would start before
// the buffer are skipped, so a short prefix leaves leading buckets at zero.
// pre, when set, supplies those sums and s must be a multiple of stride.
// targets receives one label per raw row. History sums do not use leave-one-out.
func (m *Markov) WriteWindow(data []byte, s, raw, compressed, group int, loo bool, pre *distPrefix, x []float32, targets []byte) {
	if raw < 1 || compressed < 0 || (compressed > 0 && group < 1) {
		panic("window")
	}
	span := compressed + raw
	if len(targets) != raw || len(x) != span*vocab {
		panic("window length")
	}
	clear(x[:compressed*vocab])
	if compressed > 0 {
		keep := compressed * group
		if pre != nil {
			if s%stride != 0 {
				panic("window alignment")
			}
			v := s / stride
			for i := 0; i < compressed; i++ {
				lo := v - keep + i*group
				pre.sumInto(lo, lo+group, x[i*vocab:(i+1)*vocab])
			}
		} else {
			for i := 0; i < compressed; i++ {
				row := x[i*vocab : (i+1)*vocab]
				base := -keep + i*group
				for g := 0; g < group; g++ {
					off := s + stride*(base+g)
					if off < 0 {
						continue
					}
					m.accumulate(data, off, row)
				}
			}
		}
	}
	var ctx markovKey
	base := compressed * vocab
	for j := 0; j < raw; j++ {
		off := s + stride*j
		if off < 0 || off+order > len(data) {
			panic("window past end")
		}
		copy(ctx[:], data[off:off+order])
		var tgt byte
		if off+order < len(data) {
			tgt = data[off+order]
		} else if loo {
			panic("window past end")
		}
		targets[j] = tgt
		m.Dist(ctx, tgt, loo, x[base+j*vocab:base+(j+1)*vocab])
	}
}

// accumulate adds the non-leave-one-out row at byte off onto acc.
// off+order may equal len(data); the target byte is not read.
func (m *Markov) accumulate(data []byte, off int, acc []float32) {
	if off < 0 || off+order > len(data) {
		panic("accumulate")
	}
	var ctx markovKey
	copy(ctx[:], data[off:off+order])
	var row [vocab]float32
	m.Dist(ctx, 0, false, row[:])
	for i := range acc {
		acc[i] += row[i]
	}
}
