package main

import (
	"context"
	"strings"
	"testing"
)

func TestQueryVariantsIncludesMediaSpecificQueries(t *testing.T) {
	queries := queryVariants(discoverRequest{Kind: "audiobook", Title: "Cappadonna", Creator: "Jahquel J."})
	joined := strings.Join(queries, "\n")
	if !strings.Contains(joined, "Cappadonna Jahquel J. audiobook") {
		t.Fatalf("queries did not include base audiobook query: %v", queries)
	}
	if !strings.Contains(joined, `"Cappadonna" Jahquel J. audiobook`) {
		t.Fatalf("queries did not include quoted title query: %v", queries)
	}
}

func TestOllamaModelHonorsExplicitOverride(t *testing.T) {
	s := &server{cfg: config{OllamaModel: "qwen3:14b"}}
	model, err := s.ollamaModel(context.Background())
	if err != nil || model != "qwen3:14b" {
		t.Fatalf("model=%q err=%v", model, err)
	}
}
