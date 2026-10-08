// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"math"
	"math/rand"
	"os"
	"testing"
)

func TestGutenbergBooks(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/txt-files.tar.zip"
	var tarBuf bytes.Buffer
	tw := tar.NewWriter(&tarBuf)
	write := func(hdr *tar.Header, body string) {
		t.Helper()
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if body == "" {
			return
		}
		if _, err := tw.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	write(&tar.Header{Name: "cache/epub/10000/pg10000.txt", Mode: 0644, Size: 5, Typeflag: tar.TypeReg}, "alpha")
	write(&tar.Header{Name: "notes.md", Mode: 0644, Size: 7, Typeflag: tar.TypeReg}, "skip-me")
	write(&tar.Header{Name: "link.txt", Mode: 0777, Typeflag: tar.TypeSymlink, Linkname: "pg10000.txt"}, "")
	write(&tar.Header{Name: "dir/", Mode: 0755, Typeflag: tar.TypeDir}, "")
	write(&tar.Header{Name: "cache/epub/10001/pg10001.TXT", Mode: 0644, Size: 4, Typeflag: tar.TypeReg}, "beta")
	write(&tar.Header{Name: "cache/epub/10002/pg10002.txt", Mode: 0644, Size: 5, Typeflag: tar.TypeReg}, "gamma")
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	zf, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(zf)
	w, err := zw.Create("txt-files.tar")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write(tarBuf.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := zf.Close(); err != nil {
		t.Fatal(err)
	}

	got, err := gutenbergBooks(path, 2)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "alphabeta" {
		t.Fatalf("books %q", got)
	}
	if _, err := gutenbergBooks(path, 4); err == nil {
		t.Fatal("expected a short archive to fail")
	}
}

func TestGroupSumsMatchWriteWindow(t *testing.T) {
	const raw, compressed, group = 2, 2, 2
	data := bytes.Repeat([]byte("abcdefghij"), 16)
	m := newMarkov()
	m.Train(data, len(data))
	sums := buildGroupSums(m, data, group)
	for _, s := range []int{0, 4 * stride, 8 * stride} {
		span := compressed + raw
		x := make([]float32, span*vocab)
		targets := make([]byte, raw)
		writeGroupedWindow(m, sums, data, s, raw, compressed, group, true, x, targets)
		direct := make([]float32, len(x))
		dt := make([]byte, raw)
		m.WriteWindow(data, s, raw, compressed, group, true, nil, direct, dt)
		if !bytes.Equal(targets, dt) {
			t.Fatalf("targets s=%d %q %q", s, targets, dt)
		}
		for i := range x {
			if x[i] != direct[i] {
				t.Fatalf("s=%d i=%d %g != %g", s, i, x[i], direct[i])
			}
		}
	}
}

func TestWriteGroupedWindowAlignment(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	data := bytes.Repeat([]byte("abcdefghij"), 8)
	m := newMarkov()
	m.Train(data, len(data))
	sums := buildGroupSums(m, data, 2)
	x := make([]float32, 4*vocab)
	targets := make([]byte, 2)
	writeGroupedWindow(m, sums, data, stride, 2, 2, 2, false, x, targets)
}

func TestOneShotStarts(t *testing.T) {
	const n, trainEnd, context, group, step = 100, 90, 4, 2, 4
	starts := oneShotStarts(n, trainEnd, context, group, step)
	start0, count := countWindows(n, 0, trainEnd, context, step)
	last := start0 + (count-1)*step
	if starts[0] != start0 || starts[len(starts)-1] > last {
		t.Fatalf("starts %v last %d", starts, last)
	}
	for i, s := range starts {
		if s%(group*step) != 0 {
			t.Fatalf("align %d", s)
		}
		if i > 0 && s-starts[i-1] != context*step {
			t.Fatalf("step %d", s-starts[i-1])
		}
	}
	gapped := oneShotStarts(n, trainEnd, 3, 2, step)
	for i, s := range gapped {
		if i > 0 && s-gapped[i-1] != group*step {
			t.Fatalf("group step %d", s-gapped[i-1])
		}
	}
	if names, weights := checkpointNames(0); names != markovFile || weights != weightsFile {
		t.Fatalf("default names %s %s", names, weights)
	}
	if names, weights := checkpointNames(1000); names != "markov-g1000.bin" || weights != "weights-g1000.bin" {
		t.Fatalf("gutenberg names %s %s", names, weights)
	}
}

func TestOneShotTrainFinite(t *testing.T) {
	data := bytes.Repeat([]byte("abracadabra "), 80)
	cfg := Config{Vocab: vocab, Context: 4, Compressed: 4, Group: 2, D: 8, Heads: 2, DFF: 16}
	m := newMarkov()
	trainEnd := int(float64(len(data)) * trainFraction)
	m.Train(data, trainEnd)
	sums := buildGroupSums(m, data, cfg.Group)
	starts := oneShotStarts(len(data), trainEnd, cfg.Context, cfg.Group, stride)
	if len(starts) < 3 {
		t.Fatalf("starts %d", len(starts))
	}
	w := newTensors(cfg)
	initTensors(w, cfg, rand.New(rand.NewSource(1)))
	loss := oneShotTrain(w, cfg, m, sums, data, starts[:3], 2, 1e-3)
	if math.IsNaN(loss) || math.IsInf(loss, 0) || loss <= 0 {
		t.Fatalf("loss %g", loss)
	}

	other := bytes.Repeat([]byte("xyzzzy question answer "), 40)
	m.Train(other, len(other))
	sumsB := buildGroupSums(m, other, cfg.Group)
	startsB := oneShotStarts(len(other), len(other), cfg.Context, cfg.Group, stride)
	loss = oneShotTrainSegs(w, cfg, m, []shotSeg{
		{sums: sums, data: data, starts: starts[:2]},
		{sums: sumsB, data: other, starts: startsB[:1]},
	}, 2, 1e-3)
	if math.IsNaN(loss) || math.IsInf(loss, 0) || loss <= 0 {
		t.Fatalf("two-seg loss %g", loss)
	}
}
