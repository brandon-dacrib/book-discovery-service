package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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

func TestSelectModelHonorsExplicitOverride(t *testing.T) {
	s := &server{cfg: config{OllamaModel: "qwen3:14b"}}
	model, _, err := s.selectModel(context.Background())
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

// TestParseRankingsAcceptsRealModelShapes pins the payloads observed from the
// local Ollama host. Each of these previously failed to decode, which silently
// downgraded every ranking to the deterministic fallback.
func TestParseRankingsAcceptsRealModelShapes(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    int
		first   int
	}{
		{
			name:    "bare array",
			content: `[{"index":0,"confidence":0.9,"reason":"exact"},{"index":2,"confidence":0.4,"reason":"weak"}]`,
			want:    2,
			first:   0,
		},
		{
			name:    "single object (qwen3)",
			content: `{"index":0,"confidence":0.95,"reason":"mentions audiobook"}`,
			want:    1,
			first:   0,
		},
		{
			name:    "wrapped array (gemma4)",
			content: `{"ranked_candidates":[{"index":1,"confidence":1.0,"reason":"direct link"}]}`,
			want:    1,
			first:   1,
		},
		{
			name:    "harness preamble (muse-glimmer)",
			content: ` to=user<|message|>[{"index":3,"confidence":0.97,"reason":"exact title"}]`,
			want:    1,
			first:   3,
		},
		{
			name:    "fenced array",
			content: "```json\n[{\"index\":2,\"confidence\":0.5,\"reason\":\"ok\"}]\n```",
			want:    1,
			first:   2,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ranks, err := parseRankings(tc.content)
			if err != nil {
				t.Fatalf("parseRankings(%q) error: %v", tc.content, err)
			}
			if len(ranks) != tc.want {
				t.Fatalf("got %d ranks, want %d: %+v", len(ranks), tc.want, ranks)
			}
			if ranks[0].Index != tc.first {
				t.Fatalf("first index = %d, want %d", ranks[0].Index, tc.first)
			}
		})
	}
}

func TestParseRankingsRejectsUnusableContent(t *testing.T) {
	for _, content := range []string{"", "   ", "I cannot rank these candidates."} {
		if ranks, err := parseRankings(content); err == nil {
			t.Fatalf("parseRankings(%q) unexpectedly succeeded: %+v", content, ranks)
		}
	}
}

func TestBestModelSkipsThinkingAndSpecializedModels(t *testing.T) {
	models := []ollamaModel{
		{Name: "muse-glimmer:30b-mlx", Size: 21_000_000_000, Capabilities: []string{"completion", "thinking"}},
		{Name: "qwen2.5-coder:32b", Size: 19_900_000_000, Capabilities: []string{"completion", "tools"}},
		{Name: "gemma3:27b", Size: 17_400_000_000, Capabilities: []string{"completion"}},
		{Name: "mistral-small3.2:24b", Size: 15_200_000_000, Capabilities: []string{"completion", "tools"}},
	}
	selected, ok := bestModel(models)
	if !ok {
		t.Fatal("expected a suitable model")
	}
	if selected.Name != "gemma3:27b" {
		t.Fatalf("selected %q, want gemma3:27b (largest non-thinking, non-specialized)", selected.Name)
	}
}

func TestBestModelReportsNoSuitableModel(t *testing.T) {
	models := []ollamaModel{
		{Name: "muse-glimmer:30b-mlx", Size: 21_000_000_000, Capabilities: []string{"completion", "thinking"}},
	}
	if selected, ok := bestModel(models); ok {
		t.Fatalf("expected no suitable model, got %q", selected.Name)
	}
}

func TestAuthorizedRequiresBearerToken(t *testing.T) {
	s := &server{cfg: config{ServiceToken: "s3cret"}}
	cases := []struct {
		header string
		want   bool
	}{
		{"Bearer s3cret", true},
		{"bearer s3cret", true},
		{"Bearer wrong", false},
		{"s3cret", false},
		{"", false},
		{"Basic s3cret", false},
	}
	for _, tc := range cases {
		r := httptest.NewRequest(http.MethodGet, "/v1/history", nil)
		if tc.header != "" {
			r.Header.Set("Authorization", tc.header)
		}
		if got := s.authorized(r); got != tc.want {
			t.Fatalf("authorized(%q) = %v, want %v", tc.header, got, tc.want)
		}
	}
}

func TestAuthorizedOpenWhenNoTokenConfigured(t *testing.T) {
	s := &server{cfg: config{}}
	if !s.authorized(httptest.NewRequest(http.MethodGet, "/v1/history", nil)) {
		t.Fatal("expected open access when SERVICE_API_TOKEN is unset")
	}
}

func TestDecodeJSONAcceptsCharsetParameter(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/v1/discover", strings.NewReader(`{"title":"Dune"}`))
	r.Header.Set("Content-Type", "application/json; charset=utf-8")
	var out discoverRequest
	if err := decodeJSON(r, &out); err != nil {
		t.Fatalf("decodeJSON rejected a charset parameter: %v", err)
	}
	if out.Title != "Dune" {
		t.Fatalf("title = %q, want Dune", out.Title)
	}
}

func TestDecodeJSONRejectsOtherMediaTypes(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/v1/discover", strings.NewReader("title=Dune"))
	r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	var out discoverRequest
	if err := decodeJSON(r, &out); err == nil {
		t.Fatal("expected form-encoded body to be rejected")
	}
}

func TestPromptCandidatesTrimsContentAndIndexes(t *testing.T) {
	long := strings.Repeat("a", 900)
	trimmed := promptCandidates([]candidate{
		{Title: "One", URL: "https://example.invalid/1", Content: long, Engine: "bing"},
		{Title: "Two", URL: "https://example.invalid/2"},
	})
	if len(trimmed) != 2 {
		t.Fatalf("got %d entries", len(trimmed))
	}
	if got := trimmed[0]["content"].(string); len(got) != 300 {
		t.Fatalf("content length = %d, want 300", len(got))
	}
	if trimmed[1]["index"].(int) != 1 {
		t.Fatalf("second entry index = %v, want 1", trimmed[1]["index"])
	}
	if _, ok := trimmed[0]["engine"]; ok {
		t.Fatal("engine should not be sent to the ranker")
	}
}

func TestEnvDurationFallsBackOnInvalidValues(t *testing.T) {
	t.Setenv("OLLAMA_TIMEOUT", "not-a-duration")
	if got := envDuration("OLLAMA_TIMEOUT", 42); got != 42 {
		t.Fatalf("got %v, want fallback", got)
	}
	t.Setenv("OLLAMA_TIMEOUT", "-5s")
	if got := envDuration("OLLAMA_TIMEOUT", 42); got != 42 {
		t.Fatalf("negative duration should fall back, got %v", got)
	}
	t.Setenv("OLLAMA_TIMEOUT", "90s")
	if got := envDuration("OLLAMA_TIMEOUT", 42); got.String() != "1m30s" {
		t.Fatalf("got %v, want 1m30s", got)
	}
}

// newShelfarrStub mirrors the live Shelfarr contract verified against the
// cluster: the API hangs off whatever base path SHELFARR_URL carries (this
// deployment mounts Rails at /requests), and both endpoints require a bearer
// token, answering 401 without one.
func newShelfarrStub(t *testing.T, basePath string, seen *[]string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer shelfarr-token" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		*seen = append(*seen, r.Method+" "+r.URL.Path)
		switch {
		case r.Method == http.MethodGet && r.URL.Path == basePath+"/api/v1/search":
			if r.URL.Query().Get("q") == "" {
				http.Error(w, "missing q", http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			// Mirrors the live payload: `year` is a number, and availability is
			// reported through confidence/has_*/available_book_types.
			_, _ = w.Write([]byte(`{"results":[
				{"work_id":"w-omnibus","title":"Project Hail Mary / Artemis / The Martian","author":"Andy Weir","year":2022,"source":"hardcover","source_id":"99","confidence":70,"has_audiobook":false,"has_ebook":true,"available_book_types":["ebook"]},
				{"work_id":"w-1","title":"Project Hail Mary","author":"Andy Weir","year":2021,"source":"hardcover","source_id":"42","confidence":70,"has_audiobook":true,"has_ebook":true,"available_book_types":["audiobook","ebook"]}
			]}`))
		case r.Method == http.MethodPost && r.URL.Path == basePath+"/api/v1/requests":
			var payload map[string]any
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				http.Error(w, "bad json", http.StatusBadRequest)
				return
			}
			if payload["work_id"] != "w-1" {
				http.Error(w, "missing work_id", http.StatusUnprocessableEntity)
				return
			}
			// Live Shelfarr rejects a create with "User not found" unless the
			// owning user is supplied.
			if _, ok := payload["user_id"]; !ok {
				w.WriteHeader(http.StatusNotFound)
				_, _ = w.Write([]byte(`{"errors":["User not found"]}`))
				return
			}
			w.Header().Set("Content-Type", "application/json")
			// Mirrors the live envelope, which is a list plus queue metadata.
			_, _ = w.Write([]byte(`{"requests":[{"id":"req-99","status":"searching","work_id":"w-1"}],"queued":true,"warnings":[],"errors":[]}`))
		default:
			http.Error(w, "not found", http.StatusNotFound)
		}
	}))
}

func newRequestServer(t *testing.T, shelfarrURL string) *server {
	t.Helper()
	store, err := openState(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	return &server{
		cfg: config{
			ShelfarrURL:   shelfarrURL,
			ShelfarrToken: "shelfarr-token",
			ShelfarrUser:  "1",
			Timeout:       10 * time.Second,
			OllamaTimeout: 10 * time.Second,
		},
		client: &http.Client{Timeout: 10 * time.Second},
		state:  store,
	}
}

func TestCreateShelfarrRequestResolvesWorkAndCreates(t *testing.T) {
	var seen []string
	// The live deployment sets RAILS_RELATIVE_URL_ROOT=/requests, so the base
	// path must survive into both calls.
	stub := newShelfarrStub(t, "/requests", &seen)
	defer stub.Close()
	s := newRequestServer(t, stub.URL+"/requests")

	body := strings.NewReader(`{"kind":"audiobook","title":"Project Hail Mary","creator":"Andy Weir"}`)
	r := httptest.NewRequest(http.MethodPost, "/v1/requests", body)
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.createShelfarrRequest(w, r)

	if w.Code != http.StatusAccepted {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	// The first query already contains a title match, so the broader
	// title-only query is skipped and the request is filed.
	want := []string{
		"GET /requests/api/v1/search",
		"POST /requests/api/v1/requests",
	}
	if strings.Join(seen, ",") != strings.Join(want, ",") {
		t.Fatalf("calls = %v, want %v", seen, want)
	}
	if !strings.Contains(w.Body.String(), "req-99") {
		t.Fatalf("response did not include the created request: %s", w.Body.String())
	}
}

func TestCreateShelfarrRequestIsIdempotent(t *testing.T) {
	var seen []string
	stub := newShelfarrStub(t, "/requests", &seen)
	defer stub.Close()
	s := newRequestServer(t, stub.URL+"/requests")

	post := func() *httptest.ResponseRecorder {
		r := httptest.NewRequest(http.MethodPost, "/v1/requests",
			strings.NewReader(`{"kind":"audiobook","title":"Project Hail Mary","creator":"Andy Weir"}`))
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Idempotency-Key", "retry-1")
		w := httptest.NewRecorder()
		s.createShelfarrRequest(w, r)
		return w
	}
	if code := post().Code; code != http.StatusAccepted {
		t.Fatalf("first call status = %d", code)
	}
	afterFirst := len(seen)
	second := post()
	if second.Code != http.StatusOK {
		t.Fatalf("replay status = %d, want 200", second.Code)
	}
	if !strings.Contains(second.Body.String(), "req-99") {
		t.Fatalf("replay lost the recorded response: %s", second.Body.String())
	}
	// The replay must be served from state without touching Shelfarr again.
	if len(seen) != afterFirst {
		t.Fatalf("replay issued extra upstream calls: %v", seen[afterFirst:])
	}
}

func TestCreateShelfarrRequestRejectsBadToken(t *testing.T) {
	var seen []string
	stub := newShelfarrStub(t, "/requests", &seen)
	defer stub.Close()
	s := newRequestServer(t, stub.URL+"/requests")
	s.cfg.ShelfarrToken = "wrong-token"

	r := httptest.NewRequest(http.MethodPost, "/v1/requests",
		strings.NewReader(`{"title":"Project Hail Mary"}`))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.createShelfarrRequest(w, r)

	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", w.Code)
	}
	if !strings.Contains(w.Body.String(), "401") {
		t.Fatalf("error did not surface the upstream 401: %s", w.Body.String())
	}
}

// TestCreateShelfarrRequestSurfacesInlineErrors covers Shelfarr answering 2xx
// while reporting a per-book failure in `errors`, which is what a duplicate
// active request looks like.
func TestCreateShelfarrRequestSurfacesInlineErrors(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/search") {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"results":[{"work_id":"w-1","title":"Dune","author":"Frank Herbert","year":1965,"confidence":70,"available_book_types":["audiobook"]}]}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"requests":[],"queued":false,"warnings":[],"errors":["Dune Audiobook: This audiobook already has an active request."]}`))
	}))
	defer stub.Close()
	s := newRequestServer(t, stub.URL)
	s.cfg.ShelfarrToken = ""
	s.cfg.ShelfarrURL = stub.URL

	r := httptest.NewRequest(http.MethodPost, "/v1/requests",
		strings.NewReader(`{"kind":"audiobook","title":"Dune","creator":"Frank Herbert"}`))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.cfg.ShelfarrToken = "shelfarr-token"
	s.createShelfarrRequest(w, r)

	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502 when Shelfarr reports an inline error", w.Code)
	}
	if !strings.Contains(w.Body.String(), "already has an active request") {
		t.Fatalf("error not surfaced: %s", w.Body.String())
	}
}

func TestCreateShelfarrRequestRequiresConfiguration(t *testing.T) {
	s := &server{cfg: config{}}
	r := httptest.NewRequest(http.MethodPost, "/v1/requests",
		strings.NewReader(`{"title":"Dune"}`))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.createShelfarrRequest(w, r)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 when Shelfarr is unconfigured", w.Code)
	}
}

// TestShelfarrSearchDecodesRealResponse guards against the stub drifting from
// the live API. The fixture is a verbatim response from a running Shelfarr; it
// caught `year` arriving as a JSON number while the struct declared a string,
// which made every real metadata lookup fail to decode.
func TestShelfarrSearchDecodesRealResponse(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "shelfarr_search.json"))
	if err != nil {
		t.Fatal(err)
	}
	var decoded shelfarrSearchResponse
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("live Shelfarr response does not decode: %v", err)
	}
	if len(decoded.Results) == 0 {
		t.Fatal("no results decoded")
	}
	first := decoded.Results[0]
	if first.WorkID != "hardcover:427578" {
		t.Fatalf("work_id = %q", first.WorkID)
	}
	if first.Title != "Project Hail Mary" || first.Author != "Andy Weir" {
		t.Fatalf("title/author = %q / %q", first.Title, first.Author)
	}
	if first.Year != "2021" {
		t.Fatalf("year = %q, want 2021 rendered as text", first.Year)
	}
}

// TestPickShelfarrResultPrefersRealWorkOverDerivatives reproduces the live
// failure: Shelfarr returns a study guide first and scores every row 70, so
// score alone cannot separate them.
// TestShelfarrSearchWidensOnlyWhenNeeded pins the rate-limit-conscious
// behaviour: including the author sometimes collapses recall (the live
// "Charlotte's Web E. B. White" query returned three results, none of them the
// book), so a title-only retry runs — but only when the first query came back
// without anything titled like the request.
func TestShelfarrSearchWidensOnlyWhenNeeded(t *testing.T) {
	var queries []string
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query().Get("q")
		queries = append(queries, q)
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(q, "White") {
			// Narrow query: only an omnibus, no standalone edition.
			_, _ = w.Write([]byte(`{"results":[{"work_id":"w-omni","title":"Charlotte's Web with Stuart Little","author":"E.B. White","available_book_types":["audiobook"]}]}`))
			return
		}
		_, _ = w.Write([]byte(`{"results":[{"work_id":"w-real","title":"Charlotte's Web","author":"E. B. White","available_book_types":["audiobook"]}]}`))
	}))
	defer stub.Close()
	s := &server{cfg: config{ShelfarrURL: stub.URL, ShelfarrToken: "t"}, client: stub.Client()}

	found, err := s.shelfarrSearch(context.Background(),
		createRequest{Title: "Charlotte's Web", Creator: "E. B. White"})
	if err != nil {
		t.Fatal(err)
	}
	if len(queries) != 2 {
		t.Fatalf("queries = %v, want the title-only retry", queries)
	}
	if !hasTitleMatch(found.Results, "Charlotte's Web") {
		t.Fatalf("merged results still lack the work: %+v", found.Results)
	}

	// A first query that already matches must not trigger the retry.
	queries = nil
	if _, err := s.shelfarrSearch(context.Background(),
		createRequest{Title: "Charlotte's Web"}); err != nil {
		t.Fatal(err)
	}
	if len(queries) != 1 {
		t.Fatalf("queries = %v, want a single lookup", queries)
	}
}

// TestShelfarrSearchRetriesEmptyResults covers upstream throttling, which
// presents as 200 with no results rather than a 429.
func TestShelfarrSearchRetriesEmptyResults(t *testing.T) {
	previous := shelfarrRetryDelay
	shelfarrRetryDelay = time.Millisecond
	defer func() { shelfarrRetryDelay = previous }()

	var calls int
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.Header().Set("Content-Type", "application/json")
		if calls <= 2 {
			_, _ = w.Write([]byte(`{"results":[]}`))
			return
		}
		_, _ = w.Write([]byte(`{"results":[{"work_id":"w-1","title":"Dune","author":"Frank Herbert","available_book_types":["audiobook"]}]}`))
	}))
	defer stub.Close()
	s := &server{cfg: config{ShelfarrURL: stub.URL, ShelfarrToken: "t"}, client: stub.Client()}

	found, err := s.shelfarrSearch(context.Background(),
		createRequest{Title: "Dune", Creator: "Frank Herbert"})
	if err != nil {
		t.Fatal(err)
	}
	if len(found.Results) == 0 {
		t.Fatal("retry did not recover results")
	}
	// Two empty attempts, then the retry pass succeeds on its first query.
	if calls < 3 {
		t.Fatalf("calls = %d, want a retry after the empty responses", calls)
	}
}

// TestFilterTitleMatchesRejectsNearMisses pins the shapes that reached the
// model and were chosen wrongly before this filter existed.
func TestFilterTitleMatchesRejectsNearMisses(t *testing.T) {
	input := createRequest{Title: "Charlotte's Web", Creator: "E. B. White"}
	results := []shelfarrResult{
		{WorkID: "w-omni", Title: "Charlotte's Web with Stuart Little and The Trumpet of the Swan", Author: "E.B. White"},
		{WorkID: "w-real", Title: "Charlotte's Web: The Classic Children's Story", Author: "E. B White"},
		{WorkID: "w-about", Title: "E.B. White: Some Writer! All about the author of Charlotte's Web", Author: "Beverly Gherman"},
	}
	kept := filterTitleMatches(results, input)
	if len(kept) != 1 || kept[0].WorkID != "w-real" {
		t.Fatalf("kept %+v, want only the standalone edition", kept)
	}

	// "Author : Title" catalogue form must still match.
	plato := []shelfarrResult{{WorkID: "w-p", Title: "Plato : The Republic", Author: "Plato"}}
	if got := filterTitleMatches(plato, createRequest{Title: "The Republic", Creator: "Plato"}); len(got) != 1 {
		t.Fatalf("author-prefixed title was rejected: %+v", got)
	}

	// A leading article difference is not a mismatch.
	joy := []shelfarrResult{{WorkID: "w-j", Title: "Joy of Cooking", Author: "Irma S. Rombauer"}}
	if got := filterTitleMatches(joy, createRequest{Title: "The Joy of Cooking", Creator: "Irma Rombauer"}); len(got) != 1 {
		t.Fatalf("article difference rejected the match: %+v", got)
	}

	// Nothing matching leaves the caller to fall back rather than inventing one.
	if got := filterTitleMatches(results, createRequest{Title: "Some Other Book"}); len(got) != 0 {
		t.Fatalf("expected no matches, got %+v", got)
	}

	// Merchandise borrows the title but is credited to someone else. Observed
	// live: a blank notebook outranked the novel.
	circe := []shelfarrResult{
		{WorkID: "w-merch", Title: "Circe: Madeline Miller Notebook with 8. 5 X 11 in 100 Pages", Author: "Idriss Pedro"},
		{WorkID: "w-real", Title: "Circe", Author: "Madeline Miller"},
	}
	got := filterTitleMatches(circe, createRequest{Title: "Circe", Creator: "Madeline Miller"})
	if len(got) != 1 || got[0].WorkID != "w-real" {
		t.Fatalf("kept %+v, want only the novel", got)
	}
}

// TestFilterTitleMatchesPrefersUntranslatedEdition covers a live result where
// the French "Circé" tied with the English "Circe" after accent folding.
func TestFilterTitleMatchesPrefersUntranslatedEdition(t *testing.T) {
	results := []shelfarrResult{
		{WorkID: "w-fr", Title: "Circé", Author: "Madeline Miller"},
		{WorkID: "w-en", Title: "Circe", Author: "Madeline Miller"},
	}
	got := filterTitleMatches(results, createRequest{Title: "Circe", Creator: "Madeline Miller"})
	if len(got) != 1 || got[0].WorkID != "w-en" {
		t.Fatalf("kept %+v, want the edition matching without accent folding", got)
	}
	// Asking for the translation still resolves to it.
	got = filterTitleMatches(results, createRequest{Title: "Circé", Creator: "Madeline Miller"})
	if len(got) != 1 || got[0].WorkID != "w-fr" {
		t.Fatalf("kept %+v, want the French edition", got)
	}
}

func TestAuthorMatchesToleratesCatalogueFormatting(t *testing.T) {
	for _, tc := range []struct {
		candidate, requested string
		want                 bool
	}{
		{"E.B White", "E. B. White", true},
		{"Gabriel García Márquez", "Gabriel Garcia Marquez", true}, // accents folded
		{"Irma S. Rombauer", "Irma Rombauer", true},
		{"Idriss Pedro", "Madeline Miller", false},
	} {
		if got := authorMatches(tc.candidate, tc.requested); got != tc.want {
			t.Fatalf("authorMatches(%q, %q) = %v, want %v", tc.candidate, tc.requested, got, tc.want)
		}
	}
}

func TestNormalizeTitleIgnoresArticlesAndPunctuation(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"The Joy of Cooking", "joy of cooking"},
		{"Joy of Cooking", "joy of cooking"},
		{"Salt, Fat, Acid, Heat", "salt fat acid heat"},
		{"Ender's Game", "ender s game"},
	} {
		if got := normalizeTitle(tc.in); got != tc.want {
			t.Fatalf("normalizeTitle(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestPickShelfarrResultPrefersRealWorkOverDerivatives(t *testing.T) {
	results := []shelfarrResult{
		{WorkID: "w-guide", Title: `A Study Guide for Haruki Murakami's "Kafka on the Shore"`, Author: "Gale Cengage Learning", Confidence: 70, AvailableBookTypes: []string{"audiobook"}},
		{WorkID: "w-lesson", Title: "Kafka on the Shore l Summary & Study Guide", Author: "BookRags", Confidence: 70, AvailableBookTypes: []string{"audiobook"}},
		{WorkID: "w-real", Title: "Kafka on the Shore", Author: "Haruki Murakami", Confidence: 70, AvailableBookTypes: []string{"audiobook"}},
	}
	best, err := pickShelfarrResult(results, "audiobook")
	if err != nil {
		t.Fatal(err)
	}
	if best.WorkID != "w-real" {
		t.Fatalf("picked %q (%s), want the novel", best.WorkID, best.Title)
	}
}

// TestResolveWorkUsesModelChoice covers the path that measured 8/8 live where
// first-result selection measured 4/8.
func TestResolveWorkUsesModelChoice(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"models":[{"name":"gemma3:27b","size":100,"capabilities":["completion"]}]}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		// Index 2 is the real novel; a naive pick would take index 0.
		_, _ = w.Write([]byte(`{"message":{"content":"{\"index\":2,\"confidence\":0.95,\"reason\":\"actual novel\"}"}}`))
	}))
	defer ollama.Close()

	s := &server{
		cfg:    config{OllamaURL: ollama.URL, OllamaTimeout: 10 * time.Second},
		client: ollama.Client(),
	}
	results := []shelfarrResult{
		{WorkID: "w-a", Title: "A Parade of Horribles", Confidence: 70, AvailableBookTypes: []string{"audiobook"}},
		{WorkID: "w-b", Title: "Carl's Doomsday Scenario", Confidence: 70, AvailableBookTypes: []string{"audiobook"}},
		{WorkID: "w-real", Title: "Dungeon Crawler Carl", Confidence: 70, AvailableBookTypes: []string{"audiobook"}},
	}
	best, err := s.resolveWork(context.Background(),
		createRequest{Title: "Dungeon Crawler Carl", Creator: "Matt Dinniman"}, results, "audiobook")
	if err != nil {
		t.Fatal(err)
	}
	if best.WorkID != "w-real" {
		t.Fatalf("resolved to %q, want the model's choice w-real", best.WorkID)
	}
}

func TestResolveWorkFallsBackWhenModelUnavailable(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "model gone", http.StatusInternalServerError)
	}))
	defer ollama.Close()
	s := &server{
		cfg:    config{OllamaURL: ollama.URL, OllamaTimeout: 5 * time.Second},
		client: ollama.Client(),
	}
	results := []shelfarrResult{
		{WorkID: "w-guide", Title: "Study Guide: Dungeon Crawler Carl", Confidence: 90, AvailableBookTypes: []string{"audiobook"}},
		{WorkID: "w-real", Title: "Dungeon Crawler Carl", Confidence: 70, AvailableBookTypes: []string{"audiobook"}},
	}
	best, err := s.resolveWork(context.Background(),
		createRequest{Title: "Dungeon Crawler Carl"}, results, "audiobook")
	if err != nil {
		t.Fatal(err)
	}
	if best.WorkID != "w-real" {
		t.Fatalf("fallback picked %q, want the non-derivative work", best.WorkID)
	}
}

func TestPickShelfarrResultSkipsWrongFormat(t *testing.T) {
	results := []shelfarrResult{
		// Shelfarr returns omnibus and unrelated editions alongside the real
		// work; this one would win under a naive results[0].
		{WorkID: "w-omnibus", Title: "Omnibus", Confidence: 90, AvailableBookTypes: []string{"ebook"}},
		{WorkID: "w-1", Title: "Project Hail Mary", Confidence: 70, AvailableBookTypes: []string{"audiobook", "ebook"}},
	}
	best, err := pickShelfarrResult(results, "audiobook")
	if err != nil {
		t.Fatal(err)
	}
	if best.WorkID != "w-1" {
		t.Fatalf("picked %q, want the audiobook-capable work", best.WorkID)
	}
	if _, err := pickShelfarrResult(results[:1], "audiobook"); err == nil {
		t.Fatal("expected an error when no result is available as an audiobook")
	}
	if _, err := pickShelfarrResult(nil, "audiobook"); err == nil {
		t.Fatal("expected an error for an empty result set")
	}
}

func TestSearchSurfacesBackendFailure(t *testing.T) {
	searx := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "forbidden by limiter", http.StatusForbidden)
	}))
	defer searx.Close()
	s := &server{cfg: config{SearxURL: searx.URL}, client: searx.Client()}
	_, err := s.search(context.Background(), discoverRequest{Title: "Dune", Creator: "Frank Herbert"})
	if err == nil {
		t.Fatal("expected an error when every SearXNG query fails")
	}
	if !strings.Contains(err.Error(), "403") || !strings.Contains(err.Error(), "limiter") {
		t.Fatalf("error did not surface the backend cause: %v", err)
	}
}
