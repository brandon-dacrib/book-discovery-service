package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
			_, _ = w.Write([]byte(`{"results":[{"work_id":"w-1","title":"Project Hail Mary","author":"Andy Weir","year":"2021","source":"hardcover","source_id":"42"}]}`))
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
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"req-99","status":"pending","work_id":"w-1"}`))
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
	want := []string{"GET /requests/api/v1/search", "POST /requests/api/v1/requests"}
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
	second := post()
	if second.Code != http.StatusOK {
		t.Fatalf("replay status = %d, want 200", second.Code)
	}
	if !strings.Contains(second.Body.String(), "req-99") {
		t.Fatalf("replay lost the recorded response: %s", second.Body.String())
	}
	// Two calls total: the replay must not reach Shelfarr again.
	if len(seen) != 2 {
		t.Fatalf("upstream calls = %v, want the replay served from state", seen)
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
