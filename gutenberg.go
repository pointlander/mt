// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"strings"
	"time"
)

// gutenbergBooks reads the first n regular .txt members, in archive order,
// from the tar stored inside a zip. Bytes are concatenated with no separator.
func gutenbergBooks(path string, n int) ([]byte, error) {
	if n < 1 {
		return nil, fmt.Errorf("gutenberg: need at least 1 book")
	}
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	var member *zip.File
	for _, f := range zr.File {
		if strings.HasSuffix(strings.ToLower(f.Name), ".tar") {
			member = f
			break
		}
	}
	if member == nil {
		return nil, fmt.Errorf("%s: no tar member", path)
	}
	rc, err := member.Open()
	if err != nil {
		return nil, err
	}
	defer rc.Close()

	tr := tar.NewReader(rc)
	var buf bytes.Buffer
	buf.Grow(n * 400_000)
	got := 0
	t0 := time.Now()
	for got < n {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if hdr.Typeflag != tar.TypeReg && hdr.Typeflag != tar.TypeRegA {
			continue
		}
		if !strings.HasSuffix(strings.ToLower(hdr.Name), ".txt") {
			continue
		}
		if _, err := io.Copy(&buf, tr); err != nil {
			return nil, err
		}
		got++
		if got%100 == 0 {
			fmt.Printf("gutenberg books=%d bytes=%d %s\n", got, buf.Len(), time.Since(t0).Round(time.Millisecond))
		}
	}
	if got < n {
		return nil, fmt.Errorf("%s: found %d books, need %d", path, got, n)
	}
	return buf.Bytes(), nil
}

// buildGroupSums sums non-leave-one-out rows in blocks of group, aligned to
// vector 0. Row k is added into group k/group, oldest first, matching WriteWindow.
func buildGroupSums(m *Markov, data []byte, group int) []float32 {
	if group < 1 {
		panic("group")
	}
	nvec := 0
	if len(data) > order {
		nvec = (len(data) - order) / stride
	}
	ng := 0
	if nvec > 0 {
		ng = (nvec + group - 1) / group
	}
	sums := make([]float32, ng*vocab)
	var ctx markovKey
	var row [vocab]float32
	t0 := time.Now()
	for k := 0; k < nvec; k++ {
		off := k * stride
		copy(ctx[:], data[off:off+order])
		m.Dist(ctx, 0, false, row[:])
		dst := sums[(k/group)*vocab : (k/group+1)*vocab]
		for b, v := range row {
			dst[b] += v
		}
		if (k+1)%10_000_000 == 0 {
			fmt.Printf("group sums vectors=%d/%d %s\n", k+1, nvec, time.Since(t0).Round(time.Millisecond))
		}
	}
	return sums
}

// writeGroupedWindow copies compressed buckets out of sums, then writes the
// raw rows. sums must come from buildGroupSums. The window has to open on a
// group boundary so each bucket is one stored group; a missing group is zero.
func writeGroupedWindow(m *Markov, sums []float32, data []byte, s, raw, compressed, group int, loo bool, x []float32, targets []byte) {
	if raw < 1 || s%stride != 0 {
		panic("window alignment")
	}
	if compressed > 0 && (group < 1 || (s/stride)%group != 0) {
		panic("window alignment")
	}
	if compressed < 0 || len(targets) != raw || len(x) != (compressed+raw)*vocab {
		panic("window length")
	}
	v := s / stride
	ng := len(sums) / vocab
	for i := 0; i < compressed; i++ {
		row := x[i*vocab : (i+1)*vocab]
		gidx := 0
		if group > 0 {
			gidx = v/group - compressed + i
		}
		if gidx < 0 || gidx >= ng {
			clear(row)
			continue
		}
		copy(row, sums[gidx*vocab:(gidx+1)*vocab])
	}
	m.WriteWindow(data, s, raw, 0, 1, loo, nil, x[compressed*vocab:], targets)
}

// oneShotStarts lists non-overlapping training windows inside [0, trainEnd).
// When context is a multiple of group the step is one raw window, so each
// training byte is a target once. Otherwise the step is one group, which
// keeps every bucket aligned with buildGroupSums.
func oneShotStarts(n, trainEnd, context, group, step int) []int {
	start0, count := countWindows(n, 0, trainEnd, context, step)
	if count < 1 {
		panic("not enough text for a context window")
	}
	align := step
	if group > 1 {
		align = group * step
	}
	start := alignUp(start0, align)
	last := start0 + (count-1)*step
	if start > last {
		panic("not enough text for a context window")
	}
	stepBy := context * step
	if group > 1 && context%group != 0 {
		stepBy = group * step
	}
	out := make([]int, 0, (last-start)/stepBy+1)
	for s := start; s <= last; s += stepBy {
		out = append(out, s)
	}
	if len(out) == 0 {
		panic("not enough text for a context window")
	}
	return out
}

// oneShotTrain takes one Adam step per batch of consecutive windows.
// Slots are filled on this goroutine, then the backward passes run together.
func oneShotTrain(w *tensors, cfg Config, m *Markov, sums []float32, data []byte, starts []int, batch int, lr float64) float64 {
	if batch < 1 || len(starts) < 1 {
		panic("one-shot batch")
	}
	opt := newAdam(w)
	workers := newWorkers(cfg, batch)
	nSteps := (len(starts) + batch - 1) / batch
	trainStart := time.Now()
	var lastLoss float64
	for step := 0; step < nSteps; step++ {
		lo := step * batch
		hi := lo + batch
		if hi > len(starts) {
			hi = len(starts)
		}
		n := hi - lo
		for i := 0; i < n; i++ {
			s := &workers.slots[i]
			writeGroupedWindow(m, sums, data, starts[lo+i], cfg.Context, cfg.Compressed, cfg.Group, true, s.x, s.targets[cfg.Compressed:])
		}
		loss, acc, gnorm := forwardSlots(w, cfg, workers, n)
		rate := learningRate(step, 10, lr)
		clipGrads(workers.grad, 1)
		opt.step(w, workers.grad, float32(rate))
		lastLoss = loss
		if step%10 == 0 || step+1 == nSteps {
			fmt.Printf("step %3d/%d loss=%.4f acc=%.4f grad=%.3f lr=%.2e %s\n",
				step+1, nSteps, loss, acc, gnorm, rate, time.Since(trainStart).Round(time.Millisecond))
		}
	}
	return lastLoss
}
