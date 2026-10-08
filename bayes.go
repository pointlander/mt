// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

// bayesKappas is the concentration grid for the Dirichlet prior.
// The fitted value is the one with the best training log loss.
var bayesKappas = []float64{0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384}

// extKey is a context longer than the Markov order. b[:n] is oldest to newest.
type extKey struct {
	n byte
	b [16]byte
}

// bayesModel predicts the next byte from a context longer than the Markov order.
// Counts of that context are the observations. The order-4 Markov distribution
// is the Dirichlet prior, so a context with no counts reproduces the Markov model.
type bayesModel struct {
	m     *Markov
	kappa float64
	nctx  int
	tab   map[extKey]*markovEntry
}

func bayesProb(prior float64, count int, kappa float64, total int) float64 {
	return (kappa*prior + float64(count)) / (kappa + float64(total))
}

func makeExtKey(data []byte, i, n int) extKey {
	var k extKey
	k.n = byte(n)
	copy(k.b[:n], data[i-n:i])
	return k
}

// fitBayes counts extended contexts on data[:end] and chooses kappa there.
// Training targets use leave-one-out on both the long context and the Markov prior.
func fitBayes(m *Markov, data []byte, end, nctx int) (*bayesModel, score) {
	if m == nil || nctx <= order || nctx > len(extKey{}.b) {
		panic("bayes")
	}
	if end > len(data) {
		end = len(data)
	}
	if end <= nctx {
		panic("not enough text for a context window")
	}
	tab := make(map[extKey]*markovEntry)
	for i := nctx; i < end; i++ {
		key := makeExtKey(data, i, nctx)
		e := tab[key]
		if e == nil {
			e = &markovEntry{}
			tab[key] = e
		}
		e.add(data[i])
	}
	model := &bayesModel{m: m, nctx: nctx, tab: tab}
	nll := make([]float64, len(bayesKappas))
	seen := end - nctx
	var prior [vocab]float32
	var count [vocab]uint32
	for i := nctx; i < end; i++ {
		total := model.fill(data, i, true, prior[:], &count)
		target := data[i]
		pt := float64(prior[target])
		ct := int(count[target])
		for k, kappa := range bayesKappas {
			nll[k] += -math.Log(bayesProb(pt, ct, kappa, int(total)))
		}
	}
	best := 0
	for k := 1; k < len(nll); k++ {
		if nll[k] < nll[best] {
			best = k
		}
	}
	model.kappa = bayesKappas[best]
	return model, score{nll: nll[best], n: seen}
}

// fill writes the Markov prior and the extended-context counts at data[i].
// loo removes one count of the target from both. count is cleared.
func (b *bayesModel) fill(data []byte, i int, loo bool, prior []float32, count *[vocab]uint32) (total uint32) {
	if i < order || i >= len(data) || len(prior) != vocab {
		panic("index")
	}
	var ctx markovKey
	copy(ctx[:], data[i-order:i])
	b.m.Dist(ctx, data[i], loo, prior)
	clear(count[:])
	if i < b.nctx {
		return 0
	}
	e := b.tab[makeExtKey(data, i, b.nctx)]
	if e == nil {
		return 0
	}
	total = e.total
	target := data[i]
	for j, s := range e.sym {
		n := e.cnt[j]
		if loo && s == target && n > 0 {
			n--
			total--
		}
		count[s] = n
	}
	return total
}

func posteriorMode(prior []float32, count *[vocab]uint32, kappa float64) byte {
	best := byte(0)
	bestV := kappa*float64(prior[0]) + float64(count[0])
	for y := 1; y < vocab; y++ {
		v := kappa*float64(prior[y]) + float64(count[y])
		if v > bestV {
			bestV = v
			best = byte(y)
		}
	}
	return best
}

// Evaluate scores every target in [lo, hi). Extended counts come from training.
// loo applies to the Markov prior and, when the long context was counted, to that too.
func (b *bayesModel) Evaluate(data []byte, lo, hi int, loo bool) score {
	if b.nctx <= order || !(b.kappa > 0) {
		panic("bayes")
	}
	if hi > len(data) {
		hi = len(data)
	}
	i0 := lo
	if i0 < order {
		i0 = order
	}
	var prior [vocab]float32
	var count [vocab]uint32
	var out score
	for i := i0; i < hi; i++ {
		total := b.fill(data, i, loo, prior[:], &count)
		target := data[i]
		p := bayesProb(float64(prior[target]), int(count[target]), b.kappa, int(total))
		out.n++
		out.nll += -math.Log(p)
		if posteriorMode(prior[:], &count, b.kappa) == target {
			out.correct++
		}
	}
	return out
}

// distribution is the posterior over the byte that follows prefix.
// A prefix shorter than the extended context uses the Markov prior alone.
func (b *bayesModel) distribution(prefix []byte) []float32 {
	if len(prefix) < order {
		panic("prompt shorter than 4 bytes")
	}
	var ctx markovKey
	copy(ctx[:], prefix[len(prefix)-order:])
	prior := make([]float32, vocab)
	b.m.Dist(ctx, 0, false, prior)
	var e *markovEntry
	if len(prefix) >= b.nctx {
		e = b.tab[makeExtKey(prefix, len(prefix), b.nctx)]
	}
	total := uint32(0)
	if e != nil {
		total = e.total
	}
	denom := b.kappa + float64(total)
	out := make([]float32, vocab)
	scale := float32(b.kappa / denom)
	for y := 0; y < vocab; y++ {
		out[y] = prior[y] * scale
	}
	if e != nil {
		inv := float32(1 / denom)
		for i, s := range e.sym {
			out[s] += float32(e.cnt[i]) * inv
		}
	}
	return out
}

// temperDist raises a distribution to 1/temperature and renormalizes.
// Temperature 1 leaves it unchanged. Values above 1 flatten it.
func temperDist(dist []float32, temperature float64) []float32 {
	if !(temperature > 0) {
		panic("temperature")
	}
	out := make([]float32, len(dist))
	if temperature == 1 {
		copy(out, dist)
		return out
	}
	inv := 1 / temperature
	var sum float64
	for i, p := range dist {
		if p <= 0 {
			continue
		}
		v := math.Pow(float64(p), inv)
		out[i] = float32(v)
		sum += v
	}
	if sum == 0 {
		panic("empty next-byte distribution")
	}
	scale := float32(1 / sum)
	for i := range out {
		out[i] *= scale
	}
	return out
}
