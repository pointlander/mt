// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"math"
	"testing"
)

func TestBayesProb(t *testing.T) {
	p := bayesProb(0.25, 0, 8, 0)
	if math.Abs(p-0.25) > 1e-12 {
		t.Fatalf("unseen %g", p)
	}
	got := bayesProb(0.25, 10, 2, 10)
	want := (2*0.25 + 10) / 12.0
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("mix %g want %g", got, want)
	}
}

func TestBayesExtendsAmbiguousContext(t *testing.T) {
	// Order 4 sees "aaab" before both X and Y. The byte before that resolves them.
	var data []byte
	for i := 0; i < 80; i++ {
		data = append(data, []byte("qaaabX")...)
		data = append(data, []byte("raaabY")...)
	}
	m := newMarkov()
	end := len(data) * 3 / 4
	m.Train(data, end)
	model, train := fitBayes(m, data, end, 6)
	if !(model.kappa > 0) || train.n < 1 || math.IsNaN(train.ce()) {
		t.Fatalf("fit kappa %g %+v", model.kappa, train)
	}
	// A later X whose 6-byte history is the training pattern.
	var pos int
	for i := end; i < len(data); i++ {
		if data[i] == 'X' && i >= 6 {
			pos = i
			break
		}
	}
	if pos == 0 {
		t.Fatal("no extended context in the holdout")
	}
	var prior [vocab]float32
	var count [vocab]uint32
	total := model.fill(data, pos, false, prior[:], &count)
	if total == 0 || count['X'] == 0 {
		t.Fatalf("missing long-context counts total=%d", total)
	}
	bayesP := bayesProb(float64(prior['X']), int(count['X']), model.kappa, int(total))
	if !(bayesP > float64(prior['X'])) {
		t.Fatalf("bayes %g markov %g", bayesP, prior['X'])
	}
	held := model.Evaluate(data, end, len(data), false)
	markov := m.Evaluate(data, end, len(data), 1, false)
	if !(held.ce() < markov.ce()) {
		t.Fatalf("held bayes %g markov %g", held.ce(), markov.ce())
	}
}

func TestBayesUnseenContextMatchesMarkov(t *testing.T) {
	data := bytes.Repeat([]byte("abracadabra "), 40)
	m := newMarkov()
	end := len(data) * 3 / 4
	m.Train(data, end)
	model := &bayesModel{m: m, kappa: 3, nctx: 8, tab: map[extKey]*markovEntry{}}
	i := end
	if i < 8 {
		t.Fatal("short")
	}
	var prior [vocab]float32
	var count [vocab]uint32
	total := model.fill(data, i, false, prior[:], &count)
	if total != 0 {
		t.Fatal("unseen context should have no counts")
	}
	got := model.distribution(data[:i])
	// No extended counts, so the posterior is the Markov prior.
	for y := 0; y < vocab; y++ {
		if math.Abs(float64(got[y]-prior[y])) > 1e-5 {
			t.Fatalf("y %d got %g prior %g", y, got[y], prior[y])
		}
	}
}

func TestTemperDistFlattens(t *testing.T) {
	dist := make([]float32, vocab)
	dist['a'] = 0.7
	dist['b'] = 0.3
	same := temperDist(dist, 1)
	if same['a'] != dist['a'] {
		t.Fatal("temperature 1")
	}
	flat := temperDist(dist, 8)
	if !(flat['a'] < dist['a'] && flat['a'] > flat['b']) {
		t.Fatalf("flat a %g b %g", flat['a'], flat['b'])
	}
}
