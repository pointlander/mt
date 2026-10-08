// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"os"
	"testing"
)

func TestSquadTrainText(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/train-v2.0.json"
	body := `{
	  "version": "v2.0",
	  "data": [
	    {
	      "title": "A",
	      "paragraphs": [
	        {
	          "context": "alpha beta",
	          "qas": [
	            {
	              "question": "what?",
	              "id": "1",
	              "answers": [{"text": "alpha", "answer_start": 0}],
	              "is_impossible": false
	            },
	            {
	              "question": "nope?",
	              "id": "2",
	              "answers": [],
	              "is_impossible": true
	            }
	          ]
	        }
	      ]
	    },
	    {
	      "title": "B",
	      "paragraphs": [
	        {
	          "context": "gamma",
	          "qas": [
	            {
	              "question": "which?",
	              "id": "3",
	              "answers": [
	                {"text": "gamma", "answer_start": 0},
	                {"text": "Gamma", "answer_start": 0}
	              ],
	              "is_impossible": false
	            }
	          ]
	        }
	      ]
	    }
	  ]
	}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got, n, err := squadTrainText(path)
	if err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Fatalf("examples %d", n)
	}
	want := "" +
		"A\n\nalpha beta\n\nQuestion: what?\nAnswer: alpha\n\n" +
		"A\n\nalpha beta\n\nQuestion: nope?\nAnswer: unanswerable\n\n" +
		"B\n\ngamma\n\nQuestion: which?\nAnswer: gamma\n\n" +
		"B\n\ngamma\n\nQuestion: which?\nAnswer: Gamma\n\n"
	if string(got) != want {
		t.Fatalf("text %q", got)
	}

	empty := dir + "/empty.json"
	if err := os.WriteFile(empty, []byte(`{"data":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := squadTrainText(empty); err == nil {
		t.Fatal("expected empty question set to fail")
	}
}

func TestSquadTrainFile(t *testing.T) {
	if _, err := os.Stat(squadFile); err != nil {
		t.Skip(err)
	}
	text, n, err := squadTrainText(squadFile)
	if err != nil {
		t.Fatal(err)
	}
	if n < 100000 || len(text) < 50<<20 {
		t.Fatalf("examples %d bytes %d", n, len(text))
	}
	if !bytes.Contains(text[:4096], []byte("Question: ")) || !bytes.Contains(text, []byte("\nAnswer: "+unanswerable+"\n")) {
		t.Fatal("rendered text is missing question or unanswerable answers")
	}
}
