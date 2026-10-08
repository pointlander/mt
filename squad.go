// Copyright 2026 The mt Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
)

const squadFile = "train-v2.0.json"

// unanswerable is the answer text used for SQuAD questions marked impossible.
const unanswerable = "unanswerable"

type squadDoc struct {
	Data []squadArticle `json:"data"`
}

type squadArticle struct {
	Title      string           `json:"title"`
	Paragraphs []squadParagraph `json:"paragraphs"`
}

type squadParagraph struct {
	Context string    `json:"context"`
	QAs     []squadQA `json:"qas"`
}

type squadQA struct {
	Question   string        `json:"question"`
	Answers    []squadAnswer `json:"answers"`
	Impossible bool          `json:"is_impossible"`
}

type squadAnswer struct {
	Text string `json:"text"`
}

// squadTrainText renders train-v2.0.json as next-byte text. Each example is
// the title, the paragraph, one question, and one answer, so the answer bytes
// follow the passage they come from. Impossible questions use unanswerable.
func squadTrainText(path string) (text []byte, examples int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	var doc squadDoc
	if err := json.NewDecoder(f).Decode(&doc); err != nil {
		return nil, 0, err
	}
	var b bytes.Buffer
	b.Grow(128 << 20)
	for _, art := range doc.Data {
		for _, p := range art.Paragraphs {
			for _, qa := range p.QAs {
				answers := qa.answerTexts()
				for _, ans := range answers {
					if art.Title != "" {
						b.WriteString(art.Title)
						b.WriteString("\n\n")
					}
					b.WriteString(p.Context)
					b.WriteString("\n\nQuestion: ")
					b.WriteString(qa.Question)
					b.WriteString("\nAnswer: ")
					b.WriteString(ans)
					b.WriteString("\n\n")
					examples++
				}
			}
		}
	}
	if examples == 0 {
		return nil, 0, fmt.Errorf("%s: no question answer pairs", path)
	}
	return b.Bytes(), examples, nil
}

func (qa squadQA) answerTexts() []string {
	if qa.Impossible || len(qa.Answers) == 0 {
		return []string{unanswerable}
	}
	out := make([]string, len(qa.Answers))
	for i, a := range qa.Answers {
		out[i] = a.Text
	}
	return out
}
