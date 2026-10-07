// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

const (
	markovFile  = "markov.bin"
	weightsFile = "weights.bin"
)

const (
	markovMagic  = "MTMK"
	weightsMagic = "MTWT"
	fileVersion  = 1
)

func fileExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

func writeAtomic(path string, write func(w io.Writer) error) error {
	dir := filepath.Dir(path)
	if dir == "" {
		dir = "."
	}
	f, err := os.CreateTemp(dir, "."+filepath.Base(path)+".*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	ok := false
	defer func() {
		f.Close()
		if !ok {
			os.Remove(tmp)
		}
	}()
	bw := bufio.NewWriter(f)
	if err := write(bw); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	ok = true
	return nil
}

func put(w io.Writer, data any) error {
	return binary.Write(w, binary.LittleEndian, data)
}

func take(r io.Reader, data any) error {
	return binary.Read(r, binary.LittleEndian, data)
}

func saveMarkov(path string, m *Markov) error {
	if m.observations < 0 {
		return fmt.Errorf("observations")
	}
	return writeAtomic(path, func(w io.Writer) error {
		if err := put(w, []byte(markovMagic)); err != nil {
			return err
		}
		if err := put(w, uint32(fileVersion)); err != nil {
			return err
		}
		if err := put(w, uint64(m.observations)); err != nil {
			return err
		}
		if err := put(w, uint64(len(m.tab))); err != nil {
			return err
		}
		for k, e := range m.tab {
			if k.n < 1 || int(k.n) > order || len(e.sym) != len(e.cnt) || len(e.sym) > vocab {
				return fmt.Errorf("context")
			}
			if err := put(w, k.n); err != nil {
				return err
			}
			if err := put(w, k.b[:k.n]); err != nil {
				return err
			}
			if err := put(w, e.total); err != nil {
				return err
			}
			if err := put(w, uint32(len(e.sym))); err != nil {
				return err
			}
			if len(e.sym) > 0 {
				if err := put(w, e.sym); err != nil {
					return err
				}
				if err := put(w, e.cnt); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

func loadMarkov(path string) (*Markov, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m, err := decodeMarkov(bufio.NewReader(f))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return m, nil
}

func decodeMarkov(r io.Reader) (*Markov, error) {
	var magic [4]byte
	if err := take(r, &magic); err != nil {
		return nil, err
	}
	if string(magic[:]) != markovMagic {
		return nil, fmt.Errorf("bad magic")
	}
	var version uint32
	if err := take(r, &version); err != nil {
		return nil, err
	}
	if version != fileVersion {
		return nil, fmt.Errorf("version %d", version)
	}
	var obs, count uint64
	if err := take(r, &obs); err != nil {
		return nil, err
	}
	if err := take(r, &count); err != nil {
		return nil, err
	}
	if obs > math.MaxInt || count > 1<<26 {
		return nil, fmt.Errorf("size")
	}
	m := &Markov{tab: make(map[ctxKey]*markovEntry, count), observations: int(obs)}
	for i := uint64(0); i < count; i++ {
		var n byte
		if err := take(r, &n); err != nil {
			return nil, err
		}
		if n < 1 || int(n) > order {
			return nil, fmt.Errorf("context length")
		}
		var k ctxKey
		k.n = n
		if err := take(r, k.b[:n]); err != nil {
			return nil, err
		}
		if _, ok := m.tab[k]; ok {
			return nil, fmt.Errorf("duplicate context")
		}
		var total, nsym uint32
		if err := take(r, &total); err != nil {
			return nil, err
		}
		if err := take(r, &nsym); err != nil {
			return nil, err
		}
		if nsym > vocab {
			return nil, fmt.Errorf("symbol count")
		}
		e := &markovEntry{total: total}
		if nsym > 0 {
			e.sym = make([]byte, nsym)
			e.cnt = make([]uint32, nsym)
			if err := take(r, e.sym); err != nil {
				return nil, err
			}
			if err := take(r, e.cnt); err != nil {
				return nil, err
			}
		}
		var sum uint64
		for _, c := range e.cnt {
			sum += uint64(c)
		}
		if sum != uint64(total) {
			return nil, fmt.Errorf("count total")
		}
		m.tab[k] = e
	}
	return m, nil
}

func saveWeights(path string, cfg Config, w *tensors) error {
	fields := []int{cfg.Vocab, cfg.Context, cfg.Compressed, cfg.Group, cfg.D, cfg.Heads, cfg.DFF}
	u := make([]uint32, len(fields))
	for i, n := range fields {
		if n < 0 || uint64(n) > math.MaxUint32 {
			return fmt.Errorf("config")
		}
		u[i] = uint32(n)
	}
	parts := w.list()
	return writeAtomic(path, func(out io.Writer) error {
		if err := put(out, []byte(weightsMagic)); err != nil {
			return err
		}
		if err := put(out, uint32(fileVersion)); err != nil {
			return err
		}
		if err := put(out, u); err != nil {
			return err
		}
		if err := put(out, uint32(len(parts))); err != nil {
			return err
		}
		for _, p := range parts {
			if len(p) > math.MaxUint32 {
				return fmt.Errorf("tensor length")
			}
			if err := put(out, uint32(len(p))); err != nil {
				return err
			}
			if err := put(out, p); err != nil {
				return err
			}
		}
		return nil
	})
}

func loadWeights(path string, cfg Config) (*tensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	w, err := decodeWeights(bufio.NewReader(f), cfg)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return w, nil
}

func decodeWeights(r io.Reader, cfg Config) (*tensors, error) {
	var magic [4]byte
	if err := take(r, &magic); err != nil {
		return nil, err
	}
	if string(magic[:]) != weightsMagic {
		return nil, fmt.Errorf("bad magic")
	}
	var version uint32
	if err := take(r, &version); err != nil {
		return nil, err
	}
	if version != fileVersion {
		return nil, fmt.Errorf("version %d", version)
	}
	var u [7]uint32
	if err := take(r, &u); err != nil {
		return nil, err
	}
	got := Config{
		Vocab: int(u[0]), Context: int(u[1]), Compressed: int(u[2]), Group: int(u[3]),
		D: int(u[4]), Heads: int(u[5]), DFF: int(u[6]),
	}
	if got != cfg {
		return nil, fmt.Errorf("config")
	}
	w := newTensors(cfg)
	parts := w.list()
	var nparts uint32
	if err := take(r, &nparts); err != nil {
		return nil, err
	}
	if int(nparts) != len(parts) {
		return nil, fmt.Errorf("tensor count")
	}
	for _, p := range parts {
		var n uint32
		if err := take(r, &n); err != nil {
			return nil, err
		}
		if int(n) != len(p) {
			return nil, fmt.Errorf("tensor length")
		}
		if err := take(r, p); err != nil {
			return nil, err
		}
	}
	return w, nil
}
