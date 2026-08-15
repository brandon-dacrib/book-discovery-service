package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode"
)

// userAgent identifies this service to SearXNG so its bot filter and logs can
// tell it apart from the default Go client string.
const userAgent = "book-discovery-service/1.0 (+internal)"

type config struct {
	ListenAddr    string
	SearxURL      string
	OllamaURL     string
	OllamaModel   string
	ShelfarrURL   string
	ShelfarrToken string
	ShelfarrUser  string
	ServiceToken  string
	StatePath     string
	Timeout       time.Duration
	OllamaTimeout time.Duration
}

func env(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func loadConfig() config {
	return config{
		ListenAddr:    env("LISTEN_ADDR", ":8080"),
		SearxURL:      strings.TrimRight(env("SEARXNG_URL", ""), "/"),
		OllamaURL:     strings.TrimRight(env("OLLAMA_URL", ""), "/"),
		OllamaModel:   strings.TrimSpace(os.Getenv("OLLAMA_MODEL")),
		ShelfarrURL:   strings.TrimRight(env("SHELFARR_URL", ""), "/"),
		ShelfarrToken: env("SHELFARR_API_TOKEN", ""),
		// Shelfarr attributes every request to a user and rejects the create
		// with "User not found" when none is supplied. The API token is not
		// itself bound to an account, so the owning user is configuration.
		ShelfarrUser: env("SHELFARR_USER_ID", "1"),
		ServiceToken: env("SERVICE_API_TOKEN", ""),
		StatePath:    env("STATE_PATH", "/data/discovery-state.json"),
		Timeout:      envDuration("REQUEST_TIMEOUT", 45*time.Second),
		// Ranking runs against a local Ollama host that may need to page a large
		// model in from disk first. A cold load costs far more than a warm one,
		// and this service backs background/overnight work, so the budget is
		// generous by default rather than tuned for interactive latency.
		OllamaTimeout: envDuration("OLLAMA_TIMEOUT", 10*time.Minute),
	}
}

func envDuration(key string, fallback time.Duration) time.Duration {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(raw)
	if err != nil || parsed <= 0 {
		log.Printf("invalid %s=%q; using %s", key, raw, fallback)
		return fallback
	}
	return parsed
}

type discoverRequest struct {
	Kind    string `json:"kind"`
	Title   string `json:"title"`
	Creator string `json:"creator"`
	Year    string `json:"year,omitempty"`
	ISBN    string `json:"isbn,omitempty"`
	Query   string `json:"query,omitempty"`
}

type createRequest struct {
	Kind     string `json:"kind"`
	Title    string `json:"title"`
	Creator  string `json:"creator"`
	Year     string `json:"year,omitempty"`
	ISBN     string `json:"isbn,omitempty"`
	WorkID   string `json:"work_id,omitempty"`
	BookType string `json:"book_type,omitempty"`
	Notes    string `json:"notes,omitempty"`
	Target   string `json:"target,omitempty"`
}

type recommendationRequest struct {
	Kind        string `json:"kind"`
	Title       string `json:"title"`
	Creator     string `json:"creator,omitempty"`
	Preferences string `json:"preferences,omitempty"`
}

type candidate struct {
	Title      string  `json:"title"`
	URL        string  `json:"url"`
	Content    string  `json:"content,omitempty"`
	Engine     string  `json:"engine,omitempty"`
	Kind       string  `json:"kind,omitempty"`
	Score      float64 `json:"score,omitempty"`
	Confidence float64 `json:"confidence,omitempty"`
	Reason     string  `json:"reason,omitempty"`
}

type searxResponse struct {
	Results []struct {
		Title    string  `json:"title"`
		URL      string  `json:"url"`
		Content  string  `json:"content"`
		Engine   string  `json:"engine"`
		Category string  `json:"category"`
		Score    float64 `json:"score"`
	} `json:"results"`
}

type ollamaResponse struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
}

type ollamaModel struct {
	Name         string   `json:"name"`
	Size         int64    `json:"size"`
	Capabilities []string `json:"capabilities"`
}

// httpClient keeps callers safe when a server is constructed without one, as
// tests and future embedders do.
func (s *server) httpClient() *http.Client {
	if s.client == nil {
		return http.DefaultClient
	}
	return s.client
}

func (m ollamaModel) hasCapability(name string) bool {
	for _, capability := range m.Capabilities {
		if strings.EqualFold(capability, name) {
			return true
		}
	}
	return false
}

// specializedModels are trained for a different job than judging whether a
// search result matches a title, creator, and media type. They are skipped
// during automatic selection but may still be named by OLLAMA_MODEL.
var specializedModels = []string{"coder", "embed", "code-", "starcoder", "whisper"}

// suitableForRanking rejects thinking models, whose reasoning preamble both
// slows ranking down and corrupts the JSON payload.
func (m ollamaModel) suitableForRanking() bool {
	if m.Name == "" || m.hasCapability("thinking") {
		return false
	}
	if len(m.Capabilities) > 0 && !m.hasCapability("completion") {
		return false
	}
	name := strings.ToLower(m.Name)
	for _, marker := range specializedModels {
		if strings.Contains(name, marker) {
			return false
		}
	}
	return true
}

// bestModel prefers the largest suitable model, since this service backs
// background work where answer quality matters more than tokens per second.
func bestModel(models []ollamaModel) (ollamaModel, bool) {
	var selected ollamaModel
	found := false
	for _, model := range models {
		if !model.suitableForRanking() {
			continue
		}
		if !found || model.Size > selected.Size || (model.Size == selected.Size && model.Name < selected.Name) {
			selected, found = model, true
		}
	}
	return selected, found
}

type ollamaModelsResponse struct {
	Models []ollamaModel `json:"models"`
}

type historyEntry struct {
	ID             string      `json:"id"`
	Intent         string      `json:"intent"`
	CreatedAt      time.Time   `json:"created_at"`
	Request        any         `json:"request"`
	Results        []candidate `json:"results,omitempty"`
	Response       any         `json:"response,omitempty"`
	IdempotencyKey string      `json:"idempotency_key,omitempty"`
}

type persistedState struct {
	Entries []historyEntry `json:"entries"`
}

type stateStore struct {
	mu      sync.Mutex
	path    string
	entries []historyEntry
}

// flexString accepts a JSON string or number. Shelfarr returns `year` as a
// number and there is no guarantee every metadata source agrees, so the field
// is decoded permissively and rendered as text.
type flexString string

func (f *flexString) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || string(data) == "null" {
		*f = ""
		return nil
	}
	if data[0] == '"' {
		var value string
		if err := json.Unmarshal(data, &value); err != nil {
			return err
		}
		*f = flexString(value)
		return nil
	}
	var number json.Number
	if err := json.Unmarshal(data, &number); err != nil {
		return err
	}
	*f = flexString(number.String())
	return nil
}

type shelfarrResult struct {
	WorkID   string     `json:"work_id"`
	Title    string     `json:"title"`
	Author   string     `json:"author"`
	Year     flexString `json:"year"`
	CoverURL string     `json:"cover_url"`
	// Confidence is Shelfarr's own match score (0-100). Used to avoid
	// requesting a wrong work when the query is ambiguous.
	Confidence         float64  `json:"confidence"`
	ContentKind        string   `json:"content_kind"`
	Source             string   `json:"source"`
	SourceID           string   `json:"source_id"`
	HasAudiobook       bool     `json:"has_audiobook"`
	HasEbook           bool     `json:"has_ebook"`
	AvailableBookTypes []string `json:"available_book_types"`
}

// supportsBookType reports whether Shelfarr knows of an edition in the format
// being requested, so an audiobook request is not filed against an
// ebook-only work.
func (r shelfarrResult) supportsBookType(bookType string) bool {
	for _, available := range r.AvailableBookTypes {
		if strings.EqualFold(available, bookType) {
			return true
		}
	}
	if len(r.AvailableBookTypes) > 0 {
		return false
	}
	switch strings.ToLower(bookType) {
	case "audiobook":
		return r.HasAudiobook
	case "ebook":
		return r.HasEbook
	}
	return true
}

type shelfarrSearchResponse struct {
	Results []shelfarrResult `json:"results"`
}

type server struct {
	cfg          config
	client       *http.Client
	ollamaClient *http.Client
	state        *stateStore
}

func main() {
	cfg := loadConfig()
	// Each backend enables a different half of the API, so the service starts
	// with whatever is configured and reports what that leaves disabled.
	// Resolution in particular is useful without a model: the deterministic
	// filters settle most requests on their own.
	if cfg.SearxURL == "" && cfg.ShelfarrURL == "" {
		log.Fatal("set SEARXNG_URL for discovery, SHELFARR_URL for resolution, or both")
	}
	if cfg.SearxURL == "" {
		log.Print("SEARXNG_URL unset; /v1/discover and /v1/recommendations are disabled")
	}
	if cfg.OllamaURL == "" {
		log.Print("OLLAMA_URL unset; ranking and work resolution use deterministic selection only")
	}
	s := &server{
		cfg:          cfg,
		client:       &http.Client{Timeout: cfg.Timeout},
		ollamaClient: &http.Client{Timeout: cfg.OllamaTimeout},
	}
	var err error
	s.state, err = openState(cfg.StatePath)
	if err != nil {
		log.Fatal(err)
	}
	if cfg.ServiceToken == "" {
		log.Print("WARNING: SERVICE_API_TOKEN is unset; /v1/requests and /v1/history are unauthenticated. " +
			"Set it, or keep this service on a trusted network only.")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /v1/capabilities", s.capabilities)
	mux.HandleFunc("GET /v1/history", s.history)
	mux.HandleFunc("POST /v1/discover", s.discover)
	mux.HandleFunc("POST /v1/recommendations", s.recommendations)
	mux.HandleFunc("POST /v1/resolve", s.resolve)
	mux.HandleFunc("POST /v1/requests", s.createShelfarrRequest)

	// Ranking can legitimately hold a connection open for minutes, so the write
	// budget tracks the Ollama budget instead of a short fixed value. The read
	// header timeout stays small to close off slow-header attacks.
	httpServer := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           securityHeaders(mux),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      cfg.Timeout + cfg.OllamaTimeout + time.Minute,
		IdleTimeout:       2 * time.Minute,
	}
	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-shutdown
		log.Print("shutting down")
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(ctx)
	}()
	log.Printf("book-discovery listening on %s", cfg.ListenAddr)
	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Cache-Control", "no-store")
		next.ServeHTTP(w, r)
	})
}
func (s *server) health(w http.ResponseWriter, _ *http.Request) {
	jsonResponse(w, http.StatusOK, map[string]any{
		"status":              "ok",
		"state":               s.cfg.StatePath,
		"shelfarr_configured": s.cfg.ShelfarrURL != "" && s.cfg.ShelfarrToken != "",
	})
}

func (s *server) capabilities(w http.ResponseWriter, _ *http.Request) {
	jsonResponse(w, http.StatusOK, map[string]any{
		"media_kinds": []string{"book", "audiobook", "ebook", "movie", "tv"},
		"operations":  []string{"discover", "recommend", "resolve", "create_request", "history"},
		"request_backends": []map[string]any{{
			"id":          "shelfarr",
			"media_kinds": []string{"book", "audiobook", "ebook"},
			"configured":  s.cfg.ShelfarrURL != "" && s.cfg.ShelfarrToken != "",
		}},
	})
}

func openState(path string) (*stateStore, error) {
	store := &stateStore{path: path}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return store, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read state: %w", err)
	}
	var state persistedState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("decode state: %w", err)
	}
	store.entries = state.Entries
	return store, nil
}

func (s *stateStore) append(entry historyEntry) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = append(s.entries, entry)
	if len(s.entries) > 500 {
		s.entries = s.entries[len(s.entries)-500:]
	}
	data, err := json.MarshalIndent(persistedState{Entries: s.entries}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o750); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(s.path), ".discovery-state-*")
	if err != nil {
		return err
	}
	temporaryName := temporary.Name()
	defer os.Remove(temporaryName)
	if err := temporary.Chmod(0o600); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(temporaryName, s.path)
}

func (s *stateStore) findByIdempotencyKey(key string) (historyEntry, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(s.entries) - 1; i >= 0; i-- {
		if s.entries[i].Intent == "request" && s.entries[i].IdempotencyKey == key {
			return s.entries[i], true
		}
	}
	return historyEntry{}, false
}

func (s *stateStore) recent(limit int) []historyEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	if limit < 1 || limit > 100 {
		limit = 50
	}
	start := len(s.entries) - limit
	if start < 0 {
		start = 0
	}
	entries := append([]historyEntry(nil), s.entries[start:]...)
	for left, right := 0, len(entries)-1; left < right; left, right = left+1, right-1 {
		entries[left], entries[right] = entries[right], entries[left]
	}
	return entries
}

func (s *server) history(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	limit := 50
	if raw := r.URL.Query().Get("limit"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil {
			limit = parsed
		}
	}
	jsonResponse(w, http.StatusOK, map[string]any{"entries": s.state.recent(limit)})
}

func (s *server) createShelfarrRequest(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	if s.cfg.ShelfarrURL == "" || s.cfg.ShelfarrToken == "" {
		http.Error(w, "Shelfarr integration is not configured", http.StatusServiceUnavailable)
		return
	}
	var input createRequest
	if err := decodeJSON(r, &input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(input.Title) == "" {
		http.Error(w, "title is required", http.StatusBadRequest)
		return
	}
	if input.Target != "" && !strings.EqualFold(input.Target, "shelfarr") {
		http.Error(w, "unsupported request target", http.StatusBadRequest)
		return
	}
	idempotencyKey := strings.TrimSpace(r.Header.Get("Idempotency-Key"))
	if idempotencyKey != "" && s.state != nil {
		if previous, ok := s.state.findByIdempotencyKey(idempotencyKey); ok {
			jsonResponse(w, http.StatusOK, previous.Response)
			return
		}
	}
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()
	created, err := s.shelfarrCreate(ctx, input)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	s.record("request", input, nil, created, idempotencyKey)
	jsonResponse(w, http.StatusAccepted, created)
}

// resolve runs the metadata lookup and work selection without filing anything,
// so a caller can preview which work a request would target.
func (s *server) resolve(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	if s.cfg.ShelfarrURL == "" || s.cfg.ShelfarrToken == "" {
		http.Error(w, "Shelfarr integration is not configured", http.StatusServiceUnavailable)
		return
	}
	var input createRequest
	if err := decodeJSON(r, &input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(input.Title) == "" {
		http.Error(w, "title is required", http.StatusBadRequest)
		return
	}
	bookType := requestedBookType(input)
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout+s.cfg.OllamaTimeout)
	defer cancel()
	found, err := s.shelfarrSearch(ctx, input)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	best, err := s.resolveWork(ctx, input, found.Results, bookType)
	if err != nil {
		http.Error(w, err.Error(), http.StatusUnprocessableEntity)
		return
	}
	jsonResponse(w, http.StatusOK, map[string]any{
		"work_id":    best.WorkID,
		"title":      best.Title,
		"author":     best.Author,
		"year":       string(best.Year),
		"book_type":  bookType,
		"cover_url":  best.CoverURL,
		"candidates": len(found.Results),
	})
}

// requestedBookType maps the caller's media kind onto Shelfarr's book types.
func requestedBookType(input createRequest) string {
	if input.BookType != "" {
		return input.BookType
	}
	if strings.EqualFold(input.Kind, "ebook") || strings.EqualFold(input.Kind, "book") {
		return "ebook"
	}
	return "audiobook"
}

func (s *server) authorized(r *http.Request) bool {
	if s.cfg.ServiceToken == "" {
		return true
	}
	header := r.Header.Get("Authorization")
	scheme, token, found := strings.Cut(header, " ")
	if !found || !strings.EqualFold(scheme, "Bearer") {
		return false
	}
	// Constant time so a caller cannot recover the token byte by byte from
	// response timing.
	return subtle.ConstantTimeCompare([]byte(strings.TrimSpace(token)), []byte(s.cfg.ServiceToken)) == 1
}

func (s *server) shelfarrCreate(ctx context.Context, input createRequest) (map[string]any, error) {
	workID := input.WorkID
	metadata := map[string]any{
		"title":           input.Title,
		"author":          input.Creator,
		"year":            input.Year,
		"content_kind":    "book",
		"external_source": "book-discovery-service",
		"notes":           input.Notes,
	}
	if input.ISBN != "" {
		metadata["notes"] = strings.TrimSpace(strings.Join([]string{input.Notes, "ISBN: " + input.ISBN}, "\n"))
	}
	bookType := requestedBookType(input)
	if workID == "" {
		resolved, err := s.shelfarrSearch(ctx, input)
		if err != nil {
			return nil, err
		}
		best, err := s.resolveWork(ctx, input, resolved.Results, bookType)
		if err != nil {
			return nil, err
		}
		workID = best.WorkID
		if best.Title != "" {
			metadata["title"] = best.Title
		}
		if best.Author != "" {
			metadata["author"] = best.Author
		}
		if best.Year != "" {
			metadata["year"] = string(best.Year)
		}
		if best.CoverURL != "" {
			metadata["cover_url"] = best.CoverURL
		}
		metadata["source_work_ids"] = []string{best.Source + ":" + best.SourceID}
	}
	payload := map[string]any{
		"work_id":    workID,
		"book_type":  bookType,
		"book_types": []string{bookType},
	}
	if s.cfg.ShelfarrUser != "" {
		if numeric, err := strconv.Atoi(s.cfg.ShelfarrUser); err == nil {
			payload["user_id"] = numeric
		} else {
			payload["user_id"] = s.cfg.ShelfarrUser
		}
	}
	for key, value := range metadata {
		payload[key] = value
	}
	created, err := s.shelfarrJSON(ctx, http.MethodPost, "/api/v1/requests", payload)
	if err != nil {
		return nil, err
	}
	// Shelfarr answers 2xx while still reporting a per-book failure in
	// `errors`, e.g. when an active request already exists for the work.
	if messages := stringsFrom(created["errors"]); len(messages) > 0 {
		return nil, fmt.Errorf("Shelfarr rejected the request: %s", strings.Join(messages, "; "))
	}
	return created, nil
}

// stringsFrom reads a JSON array of strings out of a decoded response.
func stringsFrom(value any) []string {
	items, ok := value.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		if text, ok := item.(string); ok && strings.TrimSpace(text) != "" {
			out = append(out, text)
		}
	}
	return out
}

// derivativeWork matches editions that are *about* a book rather than the book
// itself. Shelfarr's metadata source returns these inline with real works and
// scores every row identically, so they routinely sort above the actual title.
var derivativeWork = regexp.MustCompile(`(?i)study guide|lesson plan|\bsummary\b|summaries|sparknotes|bookrags|supersummary|cengage|quickread|flashbooks|teacher'?s guide|\bcliffs?notes\b|\bworkbook\b|\bcompanion\b|box set|omnibus|collection set|\bsampler\b|books \d+\s*-\s*\d+|series by .* \d+ books|book guide|` +
	// Catalogues also carry merchandise that borrows the title verbatim: blank
	// notebooks, journals, and quiz books credited to unrelated "authors".
	`\bnotebook\b|\bjournal\b|\bplanner\b|coloring book|\bdiary\b|\bcalendar\b|trivia|quiz book|\bunofficial\b|` +
	// Art prints and poster sets are catalogued under the book's exact title:
	// "Salt, Fat, Acid, Heat: a Collection of 20 Prints".
	`collection of \d+|\bprints\b|\bposter\b|\bart cards\b|` +
	// Serialized comic adaptations carry an issue number and are not the novel:
	// "Robert Jordan's Wheel of Time: Eye of the World #5" outranked the book.
	`#\s*\d+`)

// resolveWork picks the Shelfarr work a request should be filed against.
//
// This is the step that most needs judgement. Shelfarr returns the real book
// alongside study guides, academic theses, box sets, and other volumes in the
// same series, and reports the same confidence for all of them, so neither
// ordering nor score can separate them. Measured against a sample of ambiguous
// titles, taking the first result was correct half the time; asking the model
// to choose was correct every time.
//
// The model only ever picks an index from the list, so a bad answer can pick
// the wrong book but cannot invent one. When the model is unavailable the
// deterministic filter below still removes the most common derivative works.
func (s *server) resolveWork(ctx context.Context, input createRequest, results []shelfarrResult, bookType string) (shelfarrResult, error) {
	eligible := make([]shelfarrResult, 0, len(results))
	for _, result := range results {
		if result.WorkID != "" && result.supportsBookType(bookType) {
			eligible = append(eligible, result)
		}
	}
	if len(eligible) == 0 {
		return pickShelfarrResult(results, bookType)
	}
	// Narrow to candidates actually titled like the request before involving
	// the model. This removes whole classes of near-miss by construction —
	// omnibus editions ("Charlotte's Web with Stuart Little and The Trumpet of
	// the Swan"), workbooks, and other volumes in a series all carry a
	// different title — leaving the model to do the part it is good at:
	// telling genuine editions of the same work apart.
	if titled := filterTitleMatches(eligible, input); len(titled) > 0 {
		eligible = titled
	} else if allDerivative(eligible) {
		// Every match is a study guide, adaptation, or piece of merchandise,
		// which means the catalogue does not carry the work itself. Filing the
		// closest thing would acquire the wrong book, so refuse instead. This
		// is what a request for "The Eye of the World" hits: the catalogue
		// returns only numbered comic issues.
		return shelfarrResult{}, fmt.Errorf("Shelfarr returned only derivative editions for %q (study guides, adaptations, or merchandise); the work itself does not appear in its catalogue", input.Title)
	}
	if len(eligible) == 1 || s.cfg.OllamaURL == "" {
		return pickShelfarrResult(eligible, bookType)
	}
	selected, err := s.rankWorks(ctx, input, eligible)
	if err != nil {
		log.Printf("work resolution falling back to heuristics: %v", err)
		return pickShelfarrResult(eligible, bookType)
	}
	return selected, nil
}

func (s *server) rankWorks(ctx context.Context, input createRequest, results []shelfarrResult) (shelfarrResult, error) {
	candidates := make([]map[string]any, 0, len(results))
	for i, result := range results {
		candidates = append(candidates, map[string]any{
			"index":  i,
			"title":  result.Title,
			"author": result.Author,
			"year":   string(result.Year),
		})
	}
	data, err := json.Marshal(candidates)
	if err != nil {
		return shelfarrResult{}, err
	}
	// The rejection list is explicit because every category named here was
	// observed winning on a real query against this cluster's Shelfarr.
	prompt := fmt.Sprintf("Pick the ONE candidate that is the actual original work being requested.\n"+
		"REJECT: summaries, study guides, lesson plans, workbooks, companions, academic analyses or criticism, "+
		"biographies of the author, box sets, omnibus editions, and bundles listing several different titles.\n"+
		"REJECT other volumes in the same series: the requested title must match, not merely the series.\n"+
		"A subtitle after a colon is fine and usually indicates the real edition.\n"+
		"Prefer the edition credited to the requested author.\n"+
		"Requested: title=%q author=%q.\nCandidates: %s\n"+
		`Return ONLY JSON: {"index": <int>, "confidence": <0-1>, "reason": "<short>"}`,
		input.Title, input.Creator, data)

	ctx, cancel := context.WithTimeout(ctx, s.cfg.OllamaTimeout)
	defer cancel()
	ranks, err := s.chatRankings(ctx, prompt)
	if err != nil {
		return shelfarrResult{}, err
	}
	for _, rank := range ranks {
		if rank.Index >= 0 && rank.Index < len(results) {
			log.Printf("resolved %q to %q (%s)", input.Title, results[rank.Index].Title, rank.Reason)
			return results[rank.Index], nil
		}
	}
	return shelfarrResult{}, errors.New("model did not choose a candidate in range")
}

// pickShelfarrResult chooses which metadata match to file the request against.
// Taking the first hit blindly is wrong: Shelfarr returns omnibus editions and
// unrelated works alongside the real one, and a result may not exist in the
// format being asked for. Prefer results available in the requested book type,
// then Shelfarr's own confidence score.
func pickShelfarrResult(results []shelfarrResult, bookType string) (shelfarrResult, error) {
	var best shelfarrResult
	found := false
	bestDerivative := false
	for _, result := range results {
		if result.WorkID == "" || !result.supportsBookType(bookType) {
			continue
		}
		// Shelfarr scores every row identically, so confidence alone cannot
		// separate a novel from its study guide. Prefer any non-derivative
		// match before falling back to score.
		isDerivative := derivativeWork.MatchString(result.Title)
		switch {
		case !found:
		case bestDerivative && !isDerivative:
		case bestDerivative == isDerivative && result.Confidence > best.Confidence:
		default:
			continue
		}
		best, bestDerivative, found = result, isDerivative, true
	}
	if !found {
		if len(results) == 0 {
			return shelfarrResult{}, errors.New("Shelfarr metadata search found no matching work")
		}
		return shelfarrResult{}, fmt.Errorf("Shelfarr found %d works but none available as %s", len(results), bookType)
	}
	return best, nil
}

// shelfarrSearchLimit is deliberately well above the number of candidates a
// person would look at. Popular titles attract so many summaries, workbooks,
// and companions that the original work falls outside the first ten results:
// at limit=10 neither "Dune" nor "Atomic Habits" appeared at all, and both are
// present at 25. Recall, not ranking, was the binding constraint.
const shelfarrSearchLimit = "25"

// shelfarrSearch queries by title with the author and by title alone, then
// merges the two. Including the author usually sharpens the match but
// sometimes collapses it — "Charlotte's Web E. B. White" returns three results
// and none is the book, while "Charlotte's Web" returns fifteen and does.
func (s *server) shelfarrSearch(ctx context.Context, input createRequest) (shelfarrSearchResponse, error) {
	return s.shelfarrSearchWithRetry(ctx, input, true)
}

func (s *server) shelfarrSearchWithRetry(ctx context.Context, input createRequest, retry bool) (shelfarrSearchResponse, error) {
	// The third and fourth forms are escalations, reached only when the plain
	// queries come back without the work. A title whose series has a comic
	// adaptation returns nothing but numbered issues -- "The Eye of the World"
	// yields 20 of them -- while adding a disambiguating noun surfaces the
	// novel. Each query costs an upstream call, so they run only when needed.
	queries := []string{
		strings.Join([]string{input.Title, input.Creator}, " "),
		input.Title,
	}
	// A request carrying a series prefix or a subtitle ("Cradle: Unsouled",
	// "Sapiens: A Brief History of Humankind") often finds nothing as typed
	// while either half finds the work.
	for _, variant := range titleVariants(input.Title)[1:] {
		queries = append(queries, strings.Join([]string{variant, input.Creator}, " "))
	}
	queries = append(queries,
		strings.Join([]string{input.Title, input.Creator, "novel"}, " "),
		strings.Join([]string{input.Title, input.Creator, "book"}, " "))
	queries = uniqueNonEmpty(queries)
	var merged shelfarrSearchResponse
	seen := map[string]bool{}
	var lastErr error
	for _, query := range queries {
		found, err := s.shelfarrSearchOnce(ctx, query)
		if err != nil {
			lastErr = err
			continue
		}
		for _, result := range found.Results {
			key := result.WorkID
			if key == "" {
				key = result.Title + "|" + result.Author
			}
			if seen[key] {
				continue
			}
			seen[key] = true
			merged.Results = append(merged.Results, result)
		}
		// Shelfarr proxies these lookups to an upstream metadata provider that
		// rate limits, so the broader query only runs when the narrow one did
		// not already contain something titled like the request.
		if hasTitleMatch(merged.Results, input.Title) {
			break
		}
	}
	if len(merged.Results) == 0 && lastErr != nil {
		return shelfarrSearchResponse{}, lastErr
	}
	if len(merged.Results) == 0 && retry {
		// Shelfarr proxies to a metadata provider that throttles by returning
		// 200 with an empty result set, which is indistinguishable from a
		// genuine miss. Observed recovery is on the order of a minute, so one
		// bounded retry separates a throttle from "no such book".
		select {
		case <-ctx.Done():
			return merged, ctx.Err()
		case <-time.After(shelfarrRetryDelay):
		}
		log.Printf("Shelfarr metadata search returned nothing for %q; retrying once", input.Title)
		return s.shelfarrSearchWithRetry(ctx, input, false)
	}
	return merged, nil
}

// shelfarrRetryDelay is a variable so tests do not have to wait it out.
var shelfarrRetryDelay = 5 * time.Second

// hasTitleMatch reports whether some non-derivative result is plausibly the
// requested work, judged on the title alone.
func hasTitleMatch(results []shelfarrResult, title string) bool {
	for _, result := range results {
		if !derivativeWork.MatchString(result.Title) && titleMatches(result, title, "") {
			return true
		}
	}
	return false
}

// allDerivative reports whether every candidate is something other than the
// work itself.
func allDerivative(results []shelfarrResult) bool {
	if len(results) == 0 {
		return false
	}
	for _, result := range results {
		if !derivativeWork.MatchString(result.Title) {
			return false
		}
	}
	return true
}

func filterTitleMatches(results []shelfarrResult, input createRequest) []shelfarrResult {
	// A request written in Latin script is not asking for the Russian or
	// Turkish edition. Catalogues list translations under a title that
	// normalises to the same thing, so drop other scripts up front rather
	// than hoping the model notices.
	wantLatin := !hasNonLatinScript(input.Title)
	want := normalizeTitle(input.Title)
	out := make([]shelfarrResult, 0, len(results))
	for _, result := range results {
		// A candidate titled exactly as asked is what the caller wants, even if
		// it trips the derivative pattern — someone requesting "The Journal of
		// Best Practices" is not asking for a blank journal. Only titles that
		// match on the part before a subtitle are screened, which is where the
		// merchandise and study guides live.
		exact := normalizeTitle(result.Title) == want
		if !exact && derivativeWork.MatchString(result.Title) {
			continue
		}
		if wantLatin && hasNonLatinScript(result.Title) {
			continue
		}
		if titleMatches(result, input.Title, input.Creator) {
			out = append(out, result)
		}
	}
	// When the requested author is known and some candidate is credited to
	// them, drop the rest. Catalogues list merchandise and knock-offs under
	// the real title but a different author, and those survive a title test.
	if input.Creator != "" {
		byAuthor := make([]shelfarrResult, 0, len(out))
		for _, result := range out {
			if authorMatches(result.Author, input.Creator) {
				byAuthor = append(byAuthor, result)
			}
		}
		if len(byAuthor) > 0 {
			out = byAuthor
		}
	}
	// Accent folding makes a translated edition compare equal to the original
	// — a request for "Circe" matched the French "Circé" — so prefer titles
	// that match without needing the fold. Requesting a translation still
	// works, because then the folded form is what the caller typed.
	exact := make([]shelfarrResult, 0, len(out))
	for _, result := range out {
		if titleMatchesUnfolded(result, input.Title) {
			exact = append(exact, result)
		}
	}
	if len(exact) > 0 {
		return exact
	}
	return out
}

func titleMatchesUnfolded(result shelfarrResult, title string) bool {
	want := unfoldedKey(title)
	if want == "" {
		return false
	}
	return unfoldedKey(result.Title) == want ||
		unfoldedKey(strings.SplitN(result.Title, ":", 2)[0]) == want
}

// unfoldedKey normalises punctuation and case but deliberately leaves accents
// alone, so "Circe" and "Circé" stay distinguishable.
func unfoldedKey(value string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(value) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			b.WriteRune(' ')
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

// hasNonLatinScript reports whether a string carries letters from a script
// other than Latin, which marks a translated edition.
func hasNonLatinScript(value string) bool {
	for _, r := range value {
		if unicode.IsLetter(r) && !unicode.Is(unicode.Latin, r) {
			return true
		}
	}
	return false
}

// authorMatches compares on surname, which survives the initials, accents, and
// ordering differences between catalogues ("E. B. White" vs "E.B White").
func authorMatches(candidate, requested string) bool {
	fields := strings.Fields(normalizeTitle(requested))
	if len(fields) == 0 {
		return false
	}
	surname := fields[len(fields)-1]
	if len(surname) < 3 {
		return strings.Contains(normalizeTitle(candidate), normalizeTitle(requested))
	}
	return strings.Contains(normalizeTitle(candidate), surname)
}

// titleMatches compares the request against the whole title, the part before a
// subtitle, and — when a catalogue prefixes the author, as in
// "Plato : The Republic" — the part after it.
//
// The request is split the same way as the candidate. People paste the whole
// thing ("Sapiens: A Brief History of Humankind") or lead with the series
// ("Cradle: Unsouled"), and a comparison that only ever split the catalogue
// side would miss the work in both directions.
func titleMatches(result shelfarrResult, title, creator string) bool {
	for _, variant := range titleVariants(title) {
		if titleMatchesExact(result, variant, creator) {
			return true
		}
	}
	return false
}

// titleVariants returns the request title plus, when it carries a colon, each
// side of it. Order is longest-first so the most specific form wins.
func titleVariants(title string) []string {
	variants := []string{title}
	if before, after, found := strings.Cut(title, ":"); found {
		if strings.TrimSpace(before) != "" {
			variants = append(variants, before)
		}
		if strings.TrimSpace(after) != "" {
			variants = append(variants, after)
		}
	}
	return variants
}

func titleMatchesExact(result shelfarrResult, title, creator string) bool {
	want := normalizeTitle(title)
	if want == "" {
		return false
	}
	if normalizeTitle(result.Title) == want {
		return true
	}
	before, after, found := strings.Cut(result.Title, ":")
	if normalizeTitle(before) == want {
		return true
	}
	if found && creator != "" {
		fields := strings.Fields(normalizeTitle(creator))
		if len(fields) > 0 && strings.Contains(normalizeTitle(before), fields[len(fields)-1]) {
			return normalizeTitle(after) == want
		}
	}
	return false
}

// foldAccents maps the accented Latin letters that appear in author and title
// data to ASCII. Catalogues disagree on diacritics — "Gabriel García Márquez"
// against a request for "Gabriel Garcia Marquez" — and folding keeps that from
// silently defeating the author check. Done by hand to avoid pulling in
// golang.org/x/text for one comparison.
var accentFolds = strings.NewReplacer(
	"á", "a", "à", "a", "â", "a", "ä", "a", "ã", "a", "å", "a", "ā", "a",
	"é", "e", "è", "e", "ê", "e", "ë", "e", "ē", "e",
	"í", "i", "ì", "i", "î", "i", "ï", "i", "ī", "i",
	"ó", "o", "ò", "o", "ô", "o", "ö", "o", "õ", "o", "ø", "o", "ō", "o",
	"ú", "u", "ù", "u", "û", "u", "ü", "u", "ū", "u",
	"ñ", "n", "ç", "c", "ý", "y", "ÿ", "y", "š", "s", "ž", "z", "ł", "l",
	"æ", "ae", "œ", "oe", "ß", "ss", "đ", "d", "ð", "d", "þ", "th",
)

// normalizeTitle lowercases, folds accents, drops punctuation, and removes a
// leading article so "The Joy of Cooking" and "Joy of Cooking" compare equal.
//
// Letters outside Latin are kept rather than stripped. Discarding them made
// the Russian "Sapiens - история в картинки" normalise to bare "sapiens" and
// compare equal to the English edition.
func normalizeTitle(value string) string {
	folded := accentFolds.Replace(strings.ToLower(value))
	var b strings.Builder
	for _, r := range folded {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			b.WriteRune(' ')
		}
	}
	cleaned := strings.Join(strings.Fields(b.String()), " ")
	for _, article := range []string{"the ", "a ", "an "} {
		cleaned = strings.TrimPrefix(cleaned, article)
	}
	return strings.TrimSpace(cleaned)
}

func (s *server) shelfarrSearchOnce(ctx context.Context, query string) (shelfarrSearchResponse, error) {
	u, err := url.Parse(s.cfg.ShelfarrURL + "/api/v1/search")
	if err != nil {
		return shelfarrSearchResponse{}, err
	}
	values := u.Query()
	values.Set("q", strings.TrimSpace(query))
	values.Set("content_kind", "book")
	values.Set("limit", shelfarrSearchLimit)
	u.RawQuery = values.Encode()
	var decoded shelfarrSearchResponse
	if err := s.shelfarrJSONInto(ctx, http.MethodGet, u.String(), nil, &decoded); err != nil {
		return shelfarrSearchResponse{}, err
	}
	return decoded, nil
}

func (s *server) shelfarrJSON(ctx context.Context, method, path string, payload any) (map[string]any, error) {
	var decoded map[string]any
	if err := s.shelfarrJSONInto(ctx, method, s.cfg.ShelfarrURL+path, payload, &decoded); err != nil {
		return nil, err
	}
	return decoded, nil
}

func (s *server) shelfarrJSONInto(ctx context.Context, method, endpoint string, payload any, output any) error {
	var body io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		body = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, endpoint, body)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.ShelfarrToken)
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := s.httpClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 16<<10))
		return fmt.Errorf("Shelfarr returned %s: %s", resp.Status, strings.TrimSpace(string(message)))
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(output); err != nil {
		return fmt.Errorf("decode Shelfarr response: %w", err)
	}
	return nil
}

func (s *server) discover(w http.ResponseWriter, r *http.Request) {
	if s.cfg.SearxURL == "" {
		http.Error(w, "SearXNG is not configured", http.StatusServiceUnavailable)
		return
	}
	var req discoverRequest
	if err := decodeJSON(r, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Title) == "" && strings.TrimSpace(req.Query) == "" {
		http.Error(w, "title or query is required", http.StatusBadRequest)
		return
	}
	results, err := s.searchWithTimeout(r.Context(), req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	ranked, rankingErr := s.rankWithTimeout(r.Context(), req, results, false)
	if rankingErr != nil {
		log.Printf("ollama ranking unavailable: %v", rankingErr)
		ranked = fallbackRank(req, results)
	}
	s.record("discover", req, ranked, nil)
	jsonResponse(w, http.StatusOK, map[string]any{"request": req, "results": ranked, "ranked_by": map[bool]string{true: "ollama", false: "deterministic"}[rankingErr == nil]})
}

func (s *server) recommendations(w http.ResponseWriter, r *http.Request) {
	if s.cfg.SearxURL == "" {
		http.Error(w, "SearXNG is not configured", http.StatusServiceUnavailable)
		return
	}
	var input recommendationRequest
	if err := decodeJSON(r, &input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(input.Title) == "" {
		http.Error(w, "title is required", http.StatusBadRequest)
		return
	}
	req := discoverRequest{
		Kind:    input.Kind,
		Title:   input.Title,
		Creator: input.Creator,
		Query:   strings.TrimSpace(strings.Join([]string{"similar", input.Title, input.Creator, input.Preferences, "recommendations"}, " ")),
	}
	results, err := s.searchWithTimeout(r.Context(), req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	ranked, rankingErr := s.rankWithTimeout(r.Context(), req, results, true)
	if rankingErr != nil {
		log.Printf("ollama recommendation ranking unavailable: %v", rankingErr)
		ranked = fallbackRank(req, results)
	}
	s.record("recommendation", input, ranked, nil)
	jsonResponse(w, http.StatusOK, map[string]any{"request": input, "results": ranked, "ranked_by": map[bool]string{true: "ollama", false: "deterministic"}[rankingErr == nil]})
}

func (s *server) record(intent string, request any, results []candidate, response any, idempotencyKey ...string) {
	if s.state == nil {
		return
	}
	entry := historyEntry{
		ID:        fmt.Sprintf("%d", time.Now().UnixNano()),
		Intent:    intent,
		CreatedAt: time.Now().UTC(),
		Request:   request,
		Results:   results,
		Response:  response,
	}
	if len(idempotencyKey) > 0 {
		entry.IdempotencyKey = idempotencyKey[0]
	}
	if err := s.state.append(entry); err != nil {
		log.Printf("persist %s history: %v", intent, err)
	}
}

// searchWithTimeout and rankWithTimeout budget each phase separately: search
// should fail fast, while ranking may wait on a cold model load.
func (s *server) searchWithTimeout(parent context.Context, req discoverRequest) ([]candidate, error) {
	ctx, cancel := context.WithTimeout(parent, s.cfg.Timeout)
	defer cancel()
	return s.search(ctx, req)
}

func (s *server) rankWithTimeout(parent context.Context, req discoverRequest, candidates []candidate, recommendation bool) ([]candidate, error) {
	ctx, cancel := context.WithTimeout(parent, s.cfg.OllamaTimeout)
	defer cancel()
	return s.rank(ctx, req, candidates, recommendation)
}

func (s *server) search(ctx context.Context, req discoverRequest) ([]candidate, error) {
	queries := queryVariants(req)
	var wg sync.WaitGroup
	var mu sync.Mutex
	all := make([]candidate, 0)
	failures := make([]error, 0, len(queries))
	for _, query := range queries {
		query := query
		wg.Add(1)
		go func() {
			defer wg.Done()
			found, err := s.searx(ctx, query, req.Kind)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				failures = append(failures, fmt.Errorf("query %q: %w", query, err))
				return
			}
			all = append(all, found...)
		}()
	}
	wg.Wait()
	if len(all) == 0 {
		// Report why the fan-out came back empty; a silent "no candidates" hides
		// auth, TLS, and rate-limit failures that look identical from outside.
		if len(failures) > 0 {
			return nil, fmt.Errorf("searxng returned no candidates: %w", errors.Join(failures...))
		}
		return nil, errors.New("searxng returned no candidates")
	}
	return dedupe(all), nil
}

func queryVariants(req discoverRequest) []string {
	base := strings.TrimSpace(req.Query)
	if base == "" {
		base = strings.TrimSpace(strings.Join([]string{req.Title, req.Creator}, " "))
	}
	kind := strings.ToLower(strings.TrimSpace(req.Kind))
	suffix := ""
	switch kind {
	case "book", "audiobook":
		suffix = " audiobook"
	case "movie":
		suffix = " movie"
	case "tv", "show", "series":
		suffix = " tv series"
	}
	queries := []string{base + suffix}
	// Only add the phrase-match variant when there is a title to quote;
	// otherwise a query-only request sends a bare `""` to SearXNG.
	if strings.TrimSpace(req.Title) != "" {
		queries = append(queries, strings.TrimSpace(fmt.Sprintf("%q %s", req.Title, req.Creator))+suffix)
	}
	if req.ISBN != "" {
		queries = append(queries, req.ISBN)
	}
	return uniqueNonEmpty(queries)
}

func (s *server) searx(ctx context.Context, query, kind string) ([]candidate, error) {
	u, err := url.Parse(s.cfg.SearxURL + "/search")
	if err != nil {
		return nil, fmt.Errorf("parse SEARXNG_URL: %w", err)
	}
	q := u.Query()
	q.Set("q", query)
	q.Set("format", "json")
	q.Set("safesearch", "0")
	u.RawQuery = q.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Accept", "application/json")
	resp, err := s.httpClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<10))
		return nil, fmt.Errorf("searxng returned %s: %s", resp.Status, strings.TrimSpace(string(message)))
	}
	var decoded searxResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&decoded); err != nil {
		return nil, err
	}
	out := make([]candidate, 0, len(decoded.Results))
	for _, item := range decoded.Results {
		if item.URL == "" || item.Title == "" {
			continue
		}
		out = append(out, candidate{Title: item.Title, URL: item.URL, Content: item.Content, Engine: item.Engine, Kind: kind, Score: item.Score})
	}
	return out, nil
}

// chatRankings runs one JSON-mode chat completion and decodes the ranking
// payload. Both the web-result ranker and the Shelfarr work resolver go
// through here so model selection, the thinking opt-out, and the permissive
// reply parsing stay in one place.
func (s *server) chatRankings(ctx context.Context, prompt string) ([]rankEntry, error) {
	model, thinking, err := s.selectModel(ctx)
	if err != nil {
		return nil, err
	}
	payload := map[string]any{
		"model":    model,
		"stream":   false,
		"format":   "json",
		"messages": []map[string]string{{"role": "user", "content": prompt}},
		// Picking the right edition is a classification, not a creative task.
		// Sampling made repeat runs of the same book disagree with themselves,
		// which is both wrong more often and impossible to benchmark.
		"options": map[string]any{"temperature": 0, "seed": 1},
	}
	if thinking {
		// Only send `think` to models that advertise it; Ollama rejects the
		// field outright on models that do not.
		payload["think"] = false
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.OllamaURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("User-Agent", userAgent)
	started := time.Now()
	rankingClient := s.ollamaClient
	if rankingClient == nil {
		rankingClient = s.httpClient()
	}
	resp, err := rankingClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	log.Printf("ollama replied with %s in %s", model, time.Since(started).Round(time.Millisecond))
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("ollama returned %s", resp.Status)
	}
	var decoded ollamaResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&decoded); err != nil {
		return nil, err
	}
	return parseRankings(decoded.Message.Content)
}

func (s *server) rank(ctx context.Context, req discoverRequest, candidates []candidate, recommendation bool) ([]candidate, error) {
	if len(candidates) > 40 {
		candidates = candidates[:40]
	}
	data, err := json.Marshal(promptCandidates(candidates))
	if err != nil {
		return nil, err
	}
	instruction := "Prefer exact title and creator, correct media type, and authoritative pages."
	if recommendation {
		instruction = "Recommend genuinely related titles or shows, using the seed title, creator, media type, and stated preferences; reject pages that are only search-engine noise."
	}
	prompt := fmt.Sprintf("You rank search candidates for a media discovery request. Return ONLY a JSON array of objects with fields index (integer), confidence (0-1), reason (short). Include one entry per candidate you judge relevant. %s Request: kind=%q title=%q creator=%q year=%q isbn=%q. Candidates: %s", instruction, req.Kind, req.Title, req.Creator, req.Year, req.ISBN, data)
	ranks, err := s.chatRankings(ctx, prompt)
	if err != nil {
		return nil, err
	}
	out := make([]candidate, 0, len(ranks))
	for _, rank := range ranks {
		if rank.Index < 0 || rank.Index >= len(candidates) {
			continue
		}
		item := candidates[rank.Index]
		item.Confidence = clamp(rank.Confidence)
		item.Reason = rank.Reason
		out = append(out, item)
	}
	if len(out) == 0 {
		return nil, errors.New("ollama returned no usable rankings")
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Confidence > out[j].Confidence })
	return out, nil
}

type rankEntry struct {
	Index      int     `json:"index"`
	Confidence float64 `json:"confidence"`
	Reason     string  `json:"reason"`
}

// parseRankings tolerates the shapes local models actually emit under
// `format: json`, all of which were observed against this cluster's Ollama:
//   - a bare array, which is what the prompt asks for
//   - a single object, when the model ranks only one candidate (qwen3)
//   - an object wrapping the array under some key (gemma: "ranked_candidates")
//   - any of the above preceded by chat-harness tokens (muse-glimmer emits
//     " to=user<|message|>[...]")
//
// Rejecting these outright is what silently forced every request onto the
// deterministic fallback, so decoding stays permissive here.
func parseRankings(content string) ([]rankEntry, error) {
	content = strings.TrimSpace(content)
	if content == "" {
		return nil, errors.New("ollama returned an empty ranking payload")
	}
	if ranks, ok := decodeRankings([]byte(content)); ok {
		return ranks, nil
	}
	// Strip any preamble by retrying from the first plausible JSON delimiter.
	if start := strings.IndexAny(content, "[{"); start > 0 {
		if ranks, ok := decodeRankings([]byte(content[start:])); ok {
			return ranks, nil
		}
	}
	return nil, fmt.Errorf("ollama ranking was not usable JSON: %.200s", content)
}

func decodeRankings(data []byte) ([]rankEntry, bool) {
	// json.Decoder stops at the end of the first value, so trailing harness
	// tokens after a complete array or object do not break decoding.
	var ranks []rankEntry
	if err := json.NewDecoder(bytes.NewReader(data)).Decode(&ranks); err == nil && len(ranks) > 0 {
		return ranks, true
	}
	var single rankEntry
	if err := json.NewDecoder(bytes.NewReader(data)).Decode(&single); err == nil && single.Confidence > 0 {
		return []rankEntry{single}, true
	}
	var wrapper map[string]json.RawMessage
	if err := json.NewDecoder(bytes.NewReader(data)).Decode(&wrapper); err != nil {
		return nil, false
	}
	// Unwrap deterministically; map iteration order would otherwise make the
	// choice of key random when a model emits more than one array.
	keys := make([]string, 0, len(wrapper))
	for key := range wrapper {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		var nested []rankEntry
		if err := json.Unmarshal(wrapper[key], &nested); err == nil && len(nested) > 0 {
			return nested, true
		}
	}
	return nil, false
}

// selectModel resolves the ranking model and reports whether it advertises the
// thinking capability, which decides if `think: false` is safe to send.
func (s *server) selectModel(ctx context.Context) (string, bool, error) {
	if s.cfg.OllamaModel != "" {
		return s.cfg.OllamaModel, s.advertisesThinking(ctx, s.cfg.OllamaModel), nil
	}
	// /api/ps first: a model already resident answers far faster than one that
	// has to be paged in, so a warm suitable model beats a larger cold one.
	var fallback ollamaModel
	for _, endpoint := range []string{"/api/ps", "/api/tags"} {
		var models ollamaModelsResponse
		if err := s.ollamaJSON(ctx, endpoint, &models); err != nil || len(models.Models) == 0 {
			continue
		}
		if selected, ok := bestModel(models.Models); ok {
			return selected.Name, false, nil
		}
		if fallback.Name == "" {
			for _, model := range models.Models {
				if model.Name != "" && model.Size > fallback.Size {
					fallback = model
				}
			}
		}
	}
	if fallback.Name != "" {
		// Every model advertises thinking or is specialized; use the largest and
		// let parseRankings cope with the messier payload.
		log.Printf("no ideal ranking model available; falling back to %s", fallback.Name)
		return fallback.Name, fallback.hasCapability("thinking"), nil
	}
	return "", false, errors.New("Ollama has no available models; set OLLAMA_MODEL or load a model")
}

func (s *server) advertisesThinking(ctx context.Context, name string) bool {
	var models ollamaModelsResponse
	if err := s.ollamaJSON(ctx, "/api/tags", &models); err != nil {
		return false
	}
	for _, model := range models.Models {
		if model.Name == name {
			return model.hasCapability("thinking")
		}
	}
	return false
}

// promptCandidates trims each candidate to the fields the ranker actually
// needs and caps the snippet length, keeping the prompt small enough that a
// large local model is not paying to re-read search-engine boilerplate.
func promptCandidates(candidates []candidate) []map[string]any {
	out := make([]map[string]any, 0, len(candidates))
	for i, item := range candidates {
		content := item.Content
		if len(content) > 300 {
			content = content[:300]
		}
		out = append(out, map[string]any{
			"index":   i,
			"title":   item.Title,
			"url":     item.URL,
			"content": content,
		})
	}
	return out
}

func (s *server) ollamaJSON(ctx context.Context, endpoint string, output any) error {
	if s.cfg.OllamaURL == "" {
		return errors.New("OLLAMA_URL is not configured")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.cfg.OllamaURL+endpoint, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := s.httpClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("ollama returned %s", resp.Status)
	}
	return json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(output)
}

func fallbackRank(req discoverRequest, in []candidate) []candidate {
	out := append([]candidate(nil), in...)
	for i := range out {
		hay := strings.ToLower(out[i].Title + " " + out[i].Content)
		if strings.Contains(hay, strings.ToLower(req.Title)) {
			out[i].Confidence += .55
		}
		if req.Creator != "" && strings.Contains(hay, strings.ToLower(req.Creator)) {
			out[i].Confidence += .35
		}
		out[i].Confidence = clamp(out[i].Confidence)
		out[i].Reason = "title/creator text match"
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Confidence > out[j].Confidence })
	return out
}
func dedupe(in []candidate) []candidate {
	seen := map[string]bool{}
	out := make([]candidate, 0, len(in))
	for _, item := range in {
		key := strings.ToLower(item.URL)
		if !seen[key] {
			seen[key] = true
			out = append(out, item)
		}
	}
	return out
}
func uniqueNonEmpty(in []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, v := range in {
		v = strings.TrimSpace(v)
		if v != "" && !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	return out
}
func clamp(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
func decodeJSON(r *http.Request, dst any) error {
	// Compare the parsed media type so charset and boundary parameters, which
	// ordinary HTTP clients append, do not get rejected.
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || !strings.EqualFold(mediaType, "application/json") {
		return errors.New("Content-Type must be application/json")
	}
	dec := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	dec.DisallowUnknownFields()
	return dec.Decode(dst)
}
func jsonResponse(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
