package main

import (
	"context"
	"path/filepath"
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

func TestStateStorePersistsRecentEntries(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	store, err := openState(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.append(historyEntry{ID: "one", Intent: "discover"}); err != nil {
		t.Fatal(err)
	}
	reloaded, err := openState(path)
	if err != nil {
		t.Fatal(err)
	}
	entries := reloaded.recent(10)
	if len(entries) != 1 || entries[0].ID != "one" {
		t.Fatalf("unexpected entries: %+v", entries)
	}
	if err := reloaded.append(historyEntry{ID: "two", Intent: "request", IdempotencyKey: "retry-me", Response: map[string]string{"id": "42"}}); err != nil {
		t.Fatal(err)
	}
	entry, ok := reloaded.findByIdempotencyKey("retry-me")
	if !ok || entry.ID != "two" {
		t.Fatalf("idempotency lookup failed: %+v %v", entry, ok)
	}
}
