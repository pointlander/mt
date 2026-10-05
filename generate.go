// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"sort"
)

// A continuation's string probability is the transformer's autoregressive
// probability. Each factor is the next-byte distribution at that prefix.
// The inputs to that distribution are the order-4 Markov rows aligned to the
// byte being predicted. Monte Carlo tree search maximizes the sum of the
// log factors over a fixed number of new bytes.

const mctsCPuct = 1.5

type bytePrior struct {
	b byte
	p float64
}

type mctsNode struct {
	b        byte
	prior    float64
	logp     float64
	visits   int
	best     float64
	expanded bool
	children []mctsNode
}

type mctsSearch struct {
	length   int
	sims     int
	topk     int
	rng      *rand.Rand
	next     func(prefix []byte) []float32
	root     mctsNode
	best     float64
	bestText []byte
}

type continuation struct {
	greedy     []byte
	greedyLogp float64
	text       []byte
	logp       float64
}

// searchContinuation returns the highest-probability continuation found by
// greedy decoding and by MCTS. logp is the log probability of the continuation
// given the prompt, under next.
func searchContinuation(prompt []byte, length, sims, topk int, rng *rand.Rand, next func(prefix []byte) []float32) continuation {
	if length < 1 {
		panic("generation length")
	}
	if topk < 1 {
		panic("topk")
	}
	if sims < 0 {
		panic("sims")
	}
	s := &mctsSearch{
		length: length,
		sims:   sims,
		topk:   topk,
		rng:    rng,
		next:   next,
		best:   math.Inf(-1),
		root:   mctsNode{best: math.Inf(-1)},
	}
	greedy, greedyLogp := s.greedy(prompt)
	s.consider(greedy, greedyLogp)
	for i := 0; i < sims; i++ {
		s.simulate(prompt)
	}
	return continuation{
		greedy:     greedy,
		greedyLogp: greedyLogp,
		text:       append([]byte(nil), s.bestText...),
		logp:       s.best,
	}
}

func (s *mctsSearch) greedy(prompt []byte) ([]byte, float64) {
	cur := append([]byte(nil), prompt...)
	text := make([]byte, 0, s.length)
	var logp float64
	for i := 0; i < s.length; i++ {
		b, p := argmaxProb(s.next(cur))
		text = append(text, b)
		logp += math.Log(p)
		cur = append(cur, b)
	}
	return text, logp
}

func (s *mctsSearch) simulate(prompt []byte) {
	path := []*mctsNode{&s.root}
	prefix := append([]byte(nil), prompt...)
	for {
		node := path[len(path)-1]
		depth := len(path) - 1
		if depth == s.length {
			total := pathLogp(path)
			s.backup(path, total)
			s.consider(prefix[len(prompt):], total)
			return
		}
		if !node.expanded {
			s.expand(node, prefix)
			added, extra := s.rollout(prefix, s.length-depth)
			total := pathLogp(path) + extra
			full := append(append([]byte{}, prefix[len(prompt):]...), added...)
			s.backup(path, total)
			s.consider(full, total)
			return
		}
		child := s.selectChild(node)
		path = append(path, child)
		prefix = append(prefix, child.b)
	}
}

func (s *mctsSearch) expand(node *mctsNode, prefix []byte) {
	choices := topK(s.next(prefix), s.topk)
	node.children = make([]mctsNode, len(choices))
	for i, c := range choices {
		node.children[i] = mctsNode{
			b:     c.b,
			prior: c.p,
			logp:  math.Log(c.p),
			best:  math.Inf(-1),
		}
	}
	node.expanded = true
}

func (s *mctsSearch) selectChild(node *mctsNode) *mctsNode {
	bestScore := math.Inf(-1)
	pick := 0
	parentN := node.visits
	for i := range node.children {
		c := &node.children[i]
		q := 0.0
		if c.visits > 0 && !math.IsInf(c.best, -1) {
			q = c.best / float64(s.length)
		}
		u := q + mctsCPuct*c.prior*math.Sqrt(float64(parentN))/float64(1+c.visits)
		if u > bestScore {
			bestScore = u
			pick = i
		}
	}
	return &node.children[pick]
}

func (s *mctsSearch) rollout(prefix []byte, steps int) ([]byte, float64) {
	cur := append([]byte(nil), prefix...)
	added := make([]byte, 0, steps)
	var logp float64
	for i := 0; i < steps; i++ {
		choices := topK(s.next(cur), s.topk)
		c := samplePrior(choices, s.rng)
		added = append(added, c.b)
		logp += math.Log(c.p)
		cur = append(cur, c.b)
	}
	return added, logp
}

func (s *mctsSearch) backup(path []*mctsNode, total float64) {
	for _, n := range path {
		n.visits++
		if total > n.best {
			n.best = total
		}
	}
}

func (s *mctsSearch) consider(text []byte, logp float64) {
	if logp > s.best {
		s.best = logp
		s.bestText = append([]byte(nil), text...)
	}
}

func pathLogp(path []*mctsNode) float64 {
	var sum float64
	for _, n := range path[1:] {
		sum += n.logp
	}
	return sum
}

func argmaxProb(dist []float32) (byte, float64) {
	best := 0
	for i, v := range dist {
		if v > dist[best] {
			best = i
		}
	}
	return byte(best), float64(dist[best])
}

func topK(dist []float32, k int) []bytePrior {
	if k > len(dist) {
		k = len(dist)
	}
	idx := make([]int, len(dist))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool {
		if dist[idx[i]] == dist[idx[j]] {
			return idx[i] < idx[j]
		}
		return dist[idx[i]] > dist[idx[j]]
	})
	out := make([]bytePrior, 0, k)
	for i := 0; i < k && i < len(idx); i++ {
		p := float64(dist[idx[i]])
		if p == 0 {
			break
		}
		out = append(out, bytePrior{b: byte(idx[i]), p: p})
	}
	if len(out) == 0 {
		panic("empty next-byte distribution")
	}
	return out
}

func samplePrior(choices []bytePrior, rng *rand.Rand) bytePrior {
	var sum float64
	for _, c := range choices {
		sum += c.p
	}
	r := rng.Float64() * sum
	var acc float64
	for _, c := range choices {
		acc += c.p
		if r <= acc {
			return c
		}
	}
	return choices[len(choices)-1]
}

// generator caches transformer next-byte distributions for MCTS.
type generator struct {
	cfg     Config
	w       *tensors
	m       *Markov
	ws      *workspace
	x       []float32
	targets []byte
	cache   map[string][]float32
}

func newGenerator(cfg Config, w *tensors, m *Markov) *generator {
	return &generator{
		cfg:     cfg,
		w:       w,
		m:       m,
		ws:      newWorkspace(cfg, cfg.Context),
		x:       make([]float32, cfg.Context*cfg.Vocab),
		targets: make([]byte, cfg.Context),
		cache:   make(map[string][]float32),
	}
}

// Next is the transformer distribution for the byte that would follow prefix.
func (g *generator) Next(prefix []byte) []float32 {
	if len(prefix) < order {
		panic("prompt shorter than 4 bytes")
	}
	T := len(prefix) / order
	if T > g.cfg.Context {
		T = g.cfg.Context
	}
	key := string(prefix[len(prefix)-T*order:])
	if d, ok := g.cache[key]; ok {
		return d
	}
	g.m.PredictWindow(prefix, len(prefix), T, g.x[:T*vocab])
	forwardBackward(g.cfg, g.w, nil, g.ws, g.x[:T*vocab], g.targets[:T])
	probs := softmax(g.ws.logits[(T-1)*vocab : T*vocab])
	g.cache[key] = probs
	return probs
}

func (g *generator) Search(prompt []byte, length, sims, topk int, rng *rand.Rand) continuation {
	return searchContinuation(prompt, length, sims, topk, rng, g.Next)
}

func softmax(logits []float32) []float32 {
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	out := make([]float32, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v - max))
		out[i] = float32(e)
		sum += e
	}
	inv := float32(1 / sum)
	for i := range out {
		out[i] *= inv
	}
	return out
}
