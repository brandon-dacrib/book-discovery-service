package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// The corpus tests replay real Shelfarr metadata, captured once from a live
// instance, through the actual resolution path. They run offline in under a
// second and need no Shelfarr, no Ollama, and no network.
//
// This exists because resolution accuracy is not something unit tests can
// speak to: every real defect found so far (a study guide outranking the
// novel, a Russian edition matching an English request, a nonsense query
// resolving to a random book) was a whole-corpus behaviour, and three separate
// regressions were caught only by re-running the whole set after a fix.
//
// Regenerate the expectations after an intentional behaviour change:
//
//	go test ./cmd/book-discovery -run TestResolutionCorpus -update
//
// and read the diff. A change to expected.json is a change to which book gets
// acquired, so it should be reviewed as carefully as the code that caused it.
//
// The suite was mutation-tested by disabling each selection rule in turn and
// checking that the corpus notices. Four rules are load-bearing: the
// unadorned-edition preference, the requirement that some candidate match the
// title, the ambiguous-author refusal, and the author preference. Two are not:
// screening derivative editions and rejecting other scripts change nothing on
// this corpus, because a study guide or a translated title fails the title
// comparison anyway. They are kept as cheap insurance against catalogue drift,
// but a change to either will not be caught here.

var updateGolden = flag.Bool("update", false, "rewrite the corpus expectations")

type corpusBook struct {
	Title       string                      `json:"title"`
	Author      string                      `json:"author"`
	Genre       string                      `json:"genre"`
	Narrow      []shelfarrResult            `json:"narrow"`
	Broad       []shelfarrResult            `json:"broad"`
	Escalations map[string][]shelfarrResult `json:"escalations,omitempty"`
}

// corpusOutcome is the resolved work, or the refusal, for one request.
type corpusOutcome struct {
	WorkID string `json:"work_id,omitempty"`
	Title  string `json:"title,omitempty"`
	Author string `json:"author,omitempty"`
	// Refused records that resolution declined, which is the correct answer
	// when a catalogue does not carry the work.
	Refused bool `json:"refused,omitempty"`
}

// corpusServer answers metadata searches from the captured fixtures, keyed by
// the exact query the service issues, so query construction is exercised too:
// a change that stops issuing the title-only retry shows up as a miss.
func corpusServer(t *testing.T, books map[string]corpusBook) *httptest.Server {
	t.Helper()
	index := map[string][]shelfarrResult{}
	for _, book := range books {
		index[strings.ToLower(strings.TrimSpace(book.Title+" "+book.Author))] = book.Narrow
		index[strings.ToLower(strings.TrimSpace(book.Title))] = book.Broad
		for query, rows := range book.Escalations {
			index[strings.ToLower(strings.TrimSpace(query))] = rows
		}
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/api/v1/search") {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		query, _ := url.QueryUnescape(r.URL.Query().Get("q"))
		results := index[strings.ToLower(strings.TrimSpace(query))]
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(shelfarrSearchResponse{Results: results})
	}))
}

func loadCorpus(t *testing.T, name string) map[string]corpusBook {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "corpus", name))
	if err != nil {
		t.Fatal(err)
	}
	var books map[string]corpusBook
	if err := json.Unmarshal(raw, &books); err != nil {
		t.Fatalf("decode %s: %v", name, err)
	}
	if len(books) == 0 {
		t.Fatalf("%s is empty", name)
	}
	return books
}

func resolveCorpusBook(t *testing.T, s *server, book corpusBook, kind string) corpusOutcome {
	t.Helper()
	input := createRequest{Kind: kind, Title: book.Title, Creator: book.Author}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	found, err := s.shelfarrSearch(ctx, input)
	if err != nil {
		return corpusOutcome{Refused: true}
	}
	best, err := s.resolveWork(ctx, input, found.Results, requestedBookType(input))
	if err != nil {
		return corpusOutcome{Refused: true}
	}
	return corpusOutcome{WorkID: best.WorkID, Title: best.Title, Author: best.Author}
}

// TestResolutionCorpus resolves every captured book as both an ebook and an
// audiobook. Availability filtering differs per format, so scoring only one
// leaves half the behaviour untested.
func TestResolutionCorpus(t *testing.T) {
	previous := shelfarrRetryDelay
	shelfarrRetryDelay = time.Millisecond
	defer func() { shelfarrRetryDelay = previous }()

	golden := map[string]corpusOutcome{}
	goldenPath := filepath.Join("testdata", "corpus", "expected.json")
	if !*updateGolden {
		raw, err := os.ReadFile(goldenPath)
		if err != nil {
			t.Fatalf("read expectations (run with -update to create): %v", err)
		}
		if err := json.Unmarshal(raw, &golden); err != nil {
			t.Fatal(err)
		}
	}

	produced := map[string]corpusOutcome{}
	for _, corpus := range []string{"classic75.json", "heldout50.json", "adversarial.json"} {
		books := loadCorpus(t, corpus)
		stub := corpusServer(t, books)
		defer stub.Close()
		// No Ollama: resolution is deterministic and must stay that way.
		s := &server{
			cfg:    config{ShelfarrURL: stub.URL, ShelfarrToken: "corpus", Timeout: 30 * time.Second},
			client: stub.Client(),
		}
		keys := make([]string, 0, len(books))
		for key := range books {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			for _, kind := range []string{"audiobook", "ebook"} {
				name := fmt.Sprintf("%s|%s|%s", corpus, key, kind)
				produced[name] = resolveCorpusBook(t, s, books[key], kind)
			}
		}
	}

	if *updateGolden {
		encoded, err := json.MarshalIndent(produced, "", " ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, append(encoded, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d expectations to %s", len(produced), goldenPath)
		return
	}

	var drifted, refused int
	for name, want := range golden {
		got, ok := produced[name]
		if !ok {
			t.Errorf("%s: missing from this run", name)
			continue
		}
		if got.Refused {
			refused++
		}
		if got.WorkID != want.WorkID || got.Refused != want.Refused {
			drifted++
			t.Errorf("%s\n  want %s (%q)\n  got  %s (%q) refused=%v",
				name, want.WorkID, want.Title, got.WorkID, got.Title, got.Refused)
		}
	}
	for name := range produced {
		if _, ok := golden[name]; !ok {
			t.Errorf("%s: not in expectations; rerun with -update", name)
		}
	}
	t.Logf("%d resolutions, %d drifted, %d refusals", len(produced), drifted, refused)
}

// TestResolutionCorpusNeedsNoModel is the claim the deployment rests on: the
// corpus resolves identically with no Ollama configured, because nothing in
// the resolution path consults one.
func TestResolutionCorpusNeedsNoModel(t *testing.T) {
	books := loadCorpus(t, "classic75.json")
	stub := corpusServer(t, books)
	defer stub.Close()
	s := &server{
		cfg:    config{ShelfarrURL: stub.URL, ShelfarrToken: "corpus", Timeout: 30 * time.Second},
		client: stub.Client(),
	}
	if s.cfg.OllamaURL != "" {
		t.Fatal("corpus server must not be given a model endpoint")
	}
	var resolved int
	for _, book := range books {
		if out := resolveCorpusBook(t, s, book, "audiobook"); !out.Refused {
			resolved++
		}
	}
	// A regression that broke deterministic selection would collapse this.
	if resolved < len(books)-2 {
		t.Fatalf("only %d/%d resolved without a model", resolved, len(books))
	}
}
