// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
)

// Config is a single-layer causal transformer.
// Each position consumes a vocab-sized vector. Context is the number of raw
// probability rows that are scored. Compressed rows are prepended; each is
// the sum of Group earlier raw rows, so the sequence is Compressed+Context long.
type Config struct {
	Vocab      int
	Context    int
	Compressed int
	Group      int
	D          int
	Heads      int
	DFF        int
}

func defaultConfig() Config {
	return Config{
		Vocab: vocab, Context: 1000, Compressed: 1000, Group: 1000,
		D: 64, Heads: 4, DFF: 256,
	}
}

// seqLen is the transformer length: compressed sums, then the raw rows.
func (c Config) seqLen() int {
	if c.Context < 1 {
		panic("context")
	}
	if c.Compressed < 0 || (c.Compressed > 0 && c.Group < 1) {
		panic("compressed context")
	}
	return c.Compressed + c.Context
}

type tensors struct {
	InW, InB       []float32
	Pos            []float32
	Ln1W, Ln1B     []float32
	Wq, Wk, Wv, Wo []float32
	Bq, Bk, Bv, Bo []float32
	Ln2W, Ln2B     []float32
	W1, B1         []float32
	W2, B2         []float32
	LnFW, LnFB     []float32
	Wh, Bh         []float32
}

func (t *tensors) list() [][]float32 {
	return [][]float32{
		t.InW, t.InB, t.Pos,
		t.Ln1W, t.Ln1B,
		t.Wq, t.Wk, t.Wv, t.Wo,
		t.Bq, t.Bk, t.Bv, t.Bo,
		t.Ln2W, t.Ln2B,
		t.W1, t.B1, t.W2, t.B2,
		t.LnFW, t.LnFB,
		t.Wh, t.Bh,
	}
}

func (t *tensors) zero() {
	for _, p := range t.list() {
		clear(p)
	}
}

func newTensors(cfg Config) *tensors {
	if cfg.Heads < 1 || cfg.D%cfg.Heads != 0 {
		panic("model dimension must be divisible by the head count")
	}
	a := func(n int) []float32 { return make([]float32, n) }
	d, v, ctx, dff := cfg.D, cfg.Vocab, cfg.seqLen(), cfg.DFF
	return &tensors{
		InW: a(v * d), InB: a(d),
		Pos:  a(ctx * d),
		Ln1W: a(d), Ln1B: a(d),
		Wq: a(d * d), Wk: a(d * d), Wv: a(d * d), Wo: a(d * d),
		Bq: a(d), Bk: a(d), Bv: a(d), Bo: a(d),
		Ln2W: a(d), Ln2B: a(d),
		W1: a(d * dff), B1: a(dff),
		W2: a(dff * d), B2: a(d),
		LnFW: a(d), LnFB: a(d),
		Wh: a(d * v), Bh: a(v),
	}
}

func initNormal(r *rand.Rand, w []float32, std float64) {
	for i := range w {
		w[i] = float32(r.NormFloat64() * std)
	}
}

func initTensors(w *tensors, cfg Config, r *rand.Rand) {
	const std = 0.02
	initNormal(r, w.InW, std)
	initNormal(r, w.Pos, std)
	initNormal(r, w.Wq, std)
	initNormal(r, w.Wk, std)
	initNormal(r, w.Wv, std)
	initNormal(r, w.Wo, std)
	initNormal(r, w.W1, std)
	initNormal(r, w.W2, std)
	initNormal(r, w.Wh, std)
	for i := range w.Ln1W {
		w.Ln1W[i] = 1
		w.Ln2W[i] = 1
		w.LnFW[i] = 1
	}
}

func paramCount(w *tensors) int {
	n := 0
	for _, p := range w.list() {
		n += len(p)
	}
	return n
}

type workspace struct {
	T, D, H, Dh, DFF, V int
	h, ln1              []float32
	ln1Mean, ln1Rstd    []float32
	q, k, v             []float32
	scores, attn        []float32
	ctxv, ao, h2        []float32
	ln2                 []float32
	ln2Mean, ln2Rstd    []float32
	pre, hid, fo, h3    []float32
	lnf                 []float32
	lnfMean, lnfRstd    []float32
	logits              []float32

	gh, gln1, gq, gk, gv []float32
	gctx, gh2            []float32
	gln2, gpre, ghid     []float32
	gh3, glnf            []float32
	glogits              []float32
	dxScratch            []float32
}

func newWorkspace(cfg Config, T int) *workspace {
	if T < 1 || T > cfg.seqLen() {
		panic("workspace length")
	}
	d, h, dff, v := cfg.D, cfg.Heads, cfg.DFF, cfg.Vocab
	dh := d / h
	f := func(n int) []float32 { return make([]float32, n) }
	td, tv, th := T*d, T*v, h*T*T
	return &workspace{
		T: T, D: d, H: h, Dh: dh, DFF: dff, V: v,
		h: f(td), ln1: f(td), ln1Mean: f(T), ln1Rstd: f(T),
		q: f(td), k: f(td), v: f(td),
		scores: f(th), attn: f(th),
		ctxv: f(td), ao: f(td), h2: f(td),
		ln2: f(td), ln2Mean: f(T), ln2Rstd: f(T),
		pre: f(T * dff), hid: f(T * dff), fo: f(td), h3: f(td),
		lnf: f(td), lnfMean: f(T), lnfRstd: f(T),
		logits: f(tv),
		gh:     f(td), gln1: f(td), gq: f(td), gk: f(td), gv: f(td),
		gctx: f(td), gh2: f(td),
		gln2: f(td), gpre: f(T * dff), ghid: f(T * dff),
		gh3: f(td), glnf: f(td),
		glogits: f(tv), dxScratch: f(td),
	}
}

func attnForward(q, k, v []float32, T, H, Dh int, scale float32, scores, attn, ctx []float32) {
	D := H * Dh
	for h := 0; h < H; h++ {
		base := h * T * T
		for i := 0; i < T; i++ {
			qi := q[i*D+h*Dh : i*D+h*Dh+Dh]
			rowS := scores[base+i*T : base+i*T+T]
			rowA := attn[base+i*T : base+i*T+T]
			max := float32(-1e30)
			for j := 0; j <= i; j++ {
				s := dot(qi, k[j*D+h*Dh:j*D+h*Dh+Dh]) * scale
				rowS[j] = s
				if s > max {
					max = s
				}
			}
			var sum float32
			for j := 0; j <= i; j++ {
				e := exp32(rowS[j] - max)
				rowA[j] = e
				sum += e
			}
			inv := 1 / sum
			out := ctx[i*D+h*Dh : i*D+h*Dh+Dh]
			for d := 0; d < Dh; d++ {
				out[d] = 0
			}
			for j := 0; j <= i; j++ {
				a := rowA[j] * inv
				rowA[j] = a
				axpy(out, v[j*D+h*Dh:j*D+h*Dh+Dh], a)
			}
			for j := i + 1; j < T; j++ {
				rowS[j] = 0
				rowA[j] = 0
			}
		}
	}
}

func attnBackward(q, k, v, attn, dctx []float32, T, H, Dh int, scale float32, dq, dk, dv, dS []float32) {
	D := H * Dh
	clear(dq)
	clear(dk)
	clear(dv)
	for h := 0; h < H; h++ {
		base := h * T * T
		for i := 0; i < T; i++ {
			dci := dctx[i*D+h*Dh : i*D+h*Dh+Dh]
			rowA := attn[base+i*T : base+i*T+T]
			rowD := dS[base+i*T : base+i*T+T]
			var dotA float32
			for j := 0; j <= i; j++ {
				dA := dot(dci, v[j*D+h*Dh:j*D+h*Dh+Dh])
				rowD[j] = dA
				dotA += dA * rowA[j]
				axpy(dv[j*D+h*Dh:j*D+h*Dh+Dh], dci, rowA[j])
			}
			for j := 0; j <= i; j++ {
				rowD[j] = rowA[j] * (rowD[j] - dotA) * scale
			}
			for j := i + 1; j < T; j++ {
				rowD[j] = 0
			}
		}
		for i := 0; i < T; i++ {
			dqi := dq[i*D+h*Dh : i*D+h*Dh+Dh]
			qi := q[i*D+h*Dh : i*D+h*Dh+Dh]
			rowD := dS[base+i*T : base+i*T+T]
			for j := 0; j <= i; j++ {
				ds := rowD[j]
				axpy(dqi, k[j*D+h*Dh:j*D+h*Dh+Dh], ds)
				axpy(dk[j*D+h*Dh:j*D+h*Dh+Dh], qi, ds)
			}
		}
	}
}

// forwardBackward runs one causal window. x holds T rows of width V.
// Compressed rows sit at the front and are left out of the loss. Gradients
// are the derivative of the mean cross-entropy on the remaining rows and overwrite g.
// A workspace allocated for a longer window can run a shorter one.
// Positions still start at 0, matching training on a prefix of that length.
func forwardBackward(cfg Config, w, g *tensors, ws *workspace, x []float32, targets []byte) (loss float64, correct int) {
	T := len(targets)
	if T < 1 || T > ws.T || len(x) != T*ws.V {
		panic("batch shape")
	}
	if cfg.Compressed > T-1 {
		panic("loss range")
	}
	if g != nil {
		g.zero()
	}
	D, V, H, Dh, DFF := ws.D, ws.V, ws.H, ws.Dh, ws.DFF
	scale := 1 / sqrt32(float32(Dh))

	mul(x, T, V, w.InW, D, ws.h)
	addBias(ws.h, T, D, w.InB)
	for i := 0; i < T*D; i++ {
		ws.h[i] += w.Pos[i]
	}
	layernorm(ws.h, T, D, w.Ln1W, w.Ln1B, ws.ln1, ws.ln1Mean, ws.ln1Rstd)
	mul(ws.ln1, T, D, w.Wq, D, ws.q)
	addBias(ws.q, T, D, w.Bq)
	mul(ws.ln1, T, D, w.Wk, D, ws.k)
	addBias(ws.k, T, D, w.Bk)
	mul(ws.ln1, T, D, w.Wv, D, ws.v)
	addBias(ws.v, T, D, w.Bv)
	attnForward(ws.q, ws.k, ws.v, T, H, Dh, scale, ws.scores, ws.attn, ws.ctxv)
	mul(ws.ctxv, T, D, w.Wo, D, ws.ao)
	addBias(ws.ao, T, D, w.Bo)
	for i := 0; i < T*D; i++ {
		ws.h2[i] = ws.h[i] + ws.ao[i]
	}
	layernorm(ws.h2, T, D, w.Ln2W, w.Ln2B, ws.ln2, ws.ln2Mean, ws.ln2Rstd)
	mul(ws.ln2, T, D, w.W1, DFF, ws.pre)
	addBias(ws.pre, T, DFF, w.B1)
	applyGelu(ws.pre, ws.hid)
	mul(ws.hid, T, DFF, w.W2, D, ws.fo)
	addBias(ws.fo, T, D, w.B2)
	for i := 0; i < T*D; i++ {
		ws.h3[i] = ws.h2[i] + ws.fo[i]
	}
	layernorm(ws.h3, T, D, w.LnFW, w.LnFB, ws.lnf, ws.lnfMean, ws.lnfRstd)
	mul(ws.lnf, T, D, w.Wh, V, ws.logits)
	addBias(ws.logits, T, V, w.Bh)

	loss, correct = softmaxCE(ws.logits, targets, T, V, cfg.Compressed, ws.glogits)
	if g == nil {
		return loss, correct
	}

	// Head.
	mulAT(ws.lnf, T, D, ws.glogits, V, g.Wh)
	sumRows(ws.glogits, T, V, g.Bh)
	mulBT(ws.glogits, T, V, w.Wh, D, ws.glnf)
	layernormBackward(ws.glnf, ws.h3, w.LnFW, ws.lnfMean, ws.lnfRstd, T, D, ws.gh3, g.LnFW, g.LnFB)

	// Feed-forward, plus the residual into h2.
	mulAT(ws.hid, T, DFF, ws.gh3, D, g.W2)
	sumRows(ws.gh3, T, D, g.B2)
	mulBT(ws.gh3, T, D, w.W2, DFF, ws.ghid)
	applyDGelu(ws.pre, ws.ghid, ws.gpre)
	mulAT(ws.ln2, T, D, ws.gpre, DFF, g.W1)
	sumRows(ws.gpre, T, DFF, g.B1)
	mulBT(ws.gpre, T, DFF, w.W1, D, ws.gln2)
	layernormBackward(ws.gln2, ws.h2, w.Ln2W, ws.ln2Mean, ws.ln2Rstd, T, D, ws.dxScratch, g.Ln2W, g.Ln2B)
	for i := 0; i < T*D; i++ {
		ws.gh2[i] = ws.gh3[i] + ws.dxScratch[i]
	}

	// Attention, plus the residual into h.
	mulAT(ws.ctxv, T, D, ws.gh2, D, g.Wo)
	sumRows(ws.gh2, T, D, g.Bo)
	mulBT(ws.gh2, T, D, w.Wo, D, ws.gctx)
	attnBackward(ws.q, ws.k, ws.v, ws.attn, ws.gctx, T, H, Dh, scale, ws.gq, ws.gk, ws.gv, ws.scores)
	mulAT(ws.ln1, T, D, ws.gq, D, g.Wq)
	sumRows(ws.gq, T, D, g.Bq)
	mulBT(ws.gq, T, D, w.Wq, D, ws.gln1)
	mulAT(ws.ln1, T, D, ws.gk, D, g.Wk)
	sumRows(ws.gk, T, D, g.Bk)
	mulBT(ws.gk, T, D, w.Wk, D, ws.dxScratch)
	for i := 0; i < T*D; i++ {
		ws.gln1[i] += ws.dxScratch[i]
	}
	mulAT(ws.ln1, T, D, ws.gv, D, g.Wv)
	sumRows(ws.gv, T, D, g.Bv)
	mulBT(ws.gv, T, D, w.Wv, D, ws.dxScratch)
	for i := 0; i < T*D; i++ {
		ws.gln1[i] += ws.dxScratch[i]
	}
	layernormBackward(ws.gln1, ws.h, w.Ln1W, ws.ln1Mean, ws.ln1Rstd, T, D, ws.dxScratch, g.Ln1W, g.Ln1B)
	for i := 0; i < T*D; i++ {
		ws.gh[i] = ws.gh2[i] + ws.dxScratch[i]
	}

	mulAT(x, T, V, ws.gh, D, g.InW)
	sumRows(ws.gh, T, D, g.InB)
	for i := 0; i < T*D; i++ {
		g.Pos[i] += ws.gh[i]
	}
	return loss, correct
}

func softmaxCE(logits []float32, targets []byte, T, V, lossStart int, dlogits []float32) (loss float64, correct int) {
	if lossStart < 0 || lossStart >= T {
		panic("loss range")
	}
	n := T - lossStart
	invN := 1 / float32(n)
	clear(dlogits[:lossStart*V])
	for t := lossStart; t < T; t++ {
		row := logits[t*V : t*V+V]
		drow := dlogits[t*V : t*V+V]
		max := row[0]
		for _, v := range row[1:] {
			if v > max {
				max = v
			}
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v - max))
		}
		tgt := int(targets[t])
		if tgt < 0 || tgt >= V {
			panic("target out of range")
		}
		p := math.Exp(float64(row[tgt]-max)) / sum
		loss += -math.Log(p)
		best, arg := row[0], 0
		for i, v := range row {
			if v > best {
				best = v
				arg = i
			}
		}
		if arg == tgt {
			correct++
		}
		invSum := 1 / sum
		for i, v := range row {
			pi := math.Exp(float64(v-max)) * invSum
			g := float32(pi)
			if i == tgt {
				g -= 1
			}
			drow[i] = g * invN
		}
	}
	loss /= float64(n)
	return loss, correct
}

func argmaxRow(row []float32) (int, float32) {
	max := row[0]
	for _, v := range row[1:] {
		if v > max {
			max = v
		}
	}
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v - max))
	}
	best, arg := row[0], 0
	for i, v := range row {
		if v > best {
			best = v
			arg = i
		}
	}
	p := math.Exp(float64(row[arg]-max)) / sum
	return arg, float32(p)
}

type adam struct {
	m, v        [][]float32
	t           int
	b1, b2, eps float32
}

func newAdam(w *tensors) *adam {
	a := &adam{b1: 0.9, b2: 0.99, eps: 1e-8}
	for _, p := range w.list() {
		a.m = append(a.m, make([]float32, len(p)))
		a.v = append(a.v, make([]float32, len(p)))
	}
	return a
}

func (a *adam) step(w, g *tensors, lr float32) {
	a.t++
	c1 := 1 - float32(math.Pow(float64(a.b1), float64(a.t)))
	c2 := 1 - float32(math.Pow(float64(a.b2), float64(a.t)))
	ws, gs := w.list(), g.list()
	for pi := range ws {
		pw, pg, pm, pv := ws[pi], gs[pi], a.m[pi], a.v[pi]
		for i := range pw {
			gi := pg[i]
			pm[i] = a.b1*pm[i] + (1-a.b1)*gi
			pv[i] = a.b2*pv[i] + (1-a.b2)*gi*gi
			pw[i] -= lr * (pm[i] / c1) / (sqrt32(pv[i]/c2) + a.eps)
		}
	}
}

func gradNorm(g *tensors) float64 {
	var ss float64
	for _, p := range g.list() {
		for _, v := range p {
			ss += float64(v) * float64(v)
		}
	}
	return math.Sqrt(ss)
}

func clipGrads(g *tensors, maxNorm float32) float64 {
	norm := gradNorm(g)
	if norm > float64(maxNorm) && norm > 0 {
		s := float32(float64(maxNorm) / norm)
		for _, p := range g.list() {
			for i := range p {
				p[i] *= s
			}
		}
	}
	return norm
}

func addTensors(dst, src *tensors) {
	d, s := dst.list(), src.list()
	for i := range d {
		for j := range d[i] {
			d[i][j] += s[i][j]
		}
	}
}

func scaleTensors(t *tensors, s float32) {
	for _, p := range t.list() {
		for i := range p {
			p[i] *= s
		}
	}
}

func learningRate(step, warmup int, base float64) float64 {
	if step < warmup {
		return base * float64(step+1) / float64(warmup)
	}
	return base
}
