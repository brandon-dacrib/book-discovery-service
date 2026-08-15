package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type config struct {
	ListenAddr    string
	SearxURL      string
	OllamaURL     string
	OllamaModel   string
	ShelfarrURL   string
	ShelfarrToken string
	ServiceToken  string
	StatePath     string
	Timeout       time.Duration
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
		ServiceToken:  env("SERVICE_API_TOKEN", ""),
		StatePath:     env("STATE_PATH", "/data/discovery-state.json"),
		Timeout:       45 * time.Second,
	}
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
	Name string `json:"name"`
	Size int64  `json:"size"`
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

type shelfarrSearchResponse struct {
	Results []struct {
		WorkID      string `json:"work_id"`
		Title       string `json:"title"`
		Author      string `json:"author"`
		Year        string `json:"year"`
		CoverURL    string `json:"cover_url"`
		ContentKind string `json:"content_kind"`
		Source      string `json:"source"`
		SourceID    string `json:"source_id"`
	} `json:"results"`
}

type server struct {
	cfg    config
	client *http.Client
	state  *stateStore
}

func main() {
	cfg := loadConfig()
	if cfg.SearxURL == "" || cfg.OllamaURL == "" {
		log.Fatal("SEARXNG_URL and OLLAMA_URL are required")
	}
	s := &server{cfg: cfg, client: &http.Client{Timeout: cfg.Timeout}}
	var err error
	s.state, err = openState(cfg.StatePath)
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /v1/capabilities", s.capabilities)
	mux.HandleFunc("GET /v1/history", s.history)
	mux.HandleFunc("POST /v1/discover", s.discover)
	mux.HandleFunc("POST /v1/recommendations", s.recommendations)
	mux.HandleFunc("POST /v1/requests", s.createShelfarrRequest)
	log.Printf("book-discovery listening on %s", cfg.ListenAddr)
	log.Fatal(http.ListenAndServe(cfg.ListenAddr, securityHeaders(mux)))
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
		"operations":  []string{"discover", "recommend", "create_request", "history"},
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

func (s *server) authorized(r *http.Request) bool {
	if s.cfg.ServiceToken == "" {
		return true
	}
	return strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ") == s.cfg.ServiceToken
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
	if workID == "" {
		resolved, err := s.shelfarrSearch(ctx, input)
		if err != nil {
			return nil, err
		}
		if len(resolved.Results) == 0 {
			return nil, errors.New("Shelfarr metadata search found no matching work")
		}
		best := resolved.Results[0]
		workID = best.WorkID
		if workID == "" {
			return nil, errors.New("Shelfarr metadata result did not include a work_id")
		}
		if best.Title != "" {
			metadata["title"] = best.Title
		}
		if best.Author != "" {
			metadata["author"] = best.Author
		}
		if best.Year != "" {
			metadata["year"] = best.Year
		}
		if best.CoverURL != "" {
			metadata["cover_url"] = best.CoverURL
		}
		metadata["source_work_ids"] = []string{best.Source + ":" + best.SourceID}
	}
	bookType := input.BookType
	if bookType == "" {
		bookType = "audiobook"
		if strings.EqualFold(input.Kind, "ebook") || strings.EqualFold(input.Kind, "book") {
			bookType = "ebook"
		}
	}
	payload := map[string]any{
		"work_id":    workID,
		"book_type":  bookType,
		"book_types": []string{bookType},
	}
	for key, value := range metadata {
		payload[key] = value
	}
	return s.shelfarrJSON(ctx, http.MethodPost, "/api/v1/requests", payload)
}

func (s *server) shelfarrSearch(ctx context.Context, input createRequest) (shelfarrSearchResponse, error) {
	query := strings.TrimSpace(strings.Join([]string{input.Title, input.Creator}, " "))
	u, err := url.Parse(s.cfg.ShelfarrURL + "/api/v1/search")
	if err != nil {
		return shelfarrSearchResponse{}, err
	}
	values := u.Query()
	values.Set("q", query)
	values.Set("content_kind", "book")
	values.Set("limit", "10")
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
	resp, err := s.client.Do(req)
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
	var req discoverRequest
	if err := decodeJSON(r, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Title) == "" && strings.TrimSpace(req.Query) == "" {
		http.Error(w, "title or query is required", http.StatusBadRequest)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()
	results, err := s.search(ctx, req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	ranked, rankingErr := s.rank(ctx, req, results, false)
	if rankingErr != nil {
		log.Printf("ollama ranking unavailable: %v", rankingErr)
		ranked = fallbackRank(req, results)
	}
	s.record("discover", req, ranked, nil)
	jsonResponse(w, http.StatusOK, map[string]any{"request": req, "results": ranked, "ranked_by": map[bool]string{true: "ollama", false: "deterministic"}[rankingErr == nil]})
}

func (s *server) recommendations(w http.ResponseWriter, r *http.Request) {
	var input recommendationRequest
	if err := decodeJSON(r, &input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(input.Title) == "" {
		http.Error(w, "title is required", http.StatusBadRequest)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()
	req := discoverRequest{
		Kind:    input.Kind,
		Title:   input.Title,
		Creator: input.Creator,
		Query:   strings.TrimSpace(strings.Join([]string{"similar", input.Title, input.Creator, input.Preferences, "recommendations"}, " ")),
	}
	results, err := s.search(ctx, req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	ranked, rankingErr := s.rank(ctx, req, results, true)
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

func (s *server) search(ctx context.Context, req discoverRequest) ([]candidate, error) {
	queries := queryVariants(req)
	var wg sync.WaitGroup
	var mu sync.Mutex
	all := make([]candidate, 0)
	for _, query := range queries {
		query := query
		wg.Add(1)
		go func() {
			defer wg.Done()
			found, err := s.searx(ctx, query, req.Kind)
			if err == nil {
				mu.Lock()
				all = append(all, found...)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if len(all) == 0 {
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
	queries := []string{base + suffix, fmt.Sprintf("%q %s", req.Title, req.Creator) + suffix}
	if req.ISBN != "" {
		queries = append(queries, req.ISBN)
	}
	return uniqueNonEmpty(queries)
}

func (s *server) searx(ctx context.Context, query, kind string) ([]candidate, error) {
	u, _ := url.Parse(s.cfg.SearxURL + "/search")
	q := u.Query()
	q.Set("q", query)
	q.Set("format", "json")
	q.Set("safesearch", "0")
	u.RawQuery = q.Encode()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("searxng returned %s", resp.Status)
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

func (s *server) rank(ctx context.Context, req discoverRequest, candidates []candidate, recommendation bool) ([]candidate, error) {
	if len(candidates) > 40 {
		candidates = candidates[:40]
	}
	data, _ := json.Marshal(candidates)
	instruction := "Prefer exact title and creator, correct media type, and authoritative pages."
	if recommendation {
		instruction = "Recommend genuinely related titles or shows, using the seed title, creator, media type, and stated preferences; reject pages that are only search-engine noise."
	}
	prompt := fmt.Sprintf("You rank search candidates for a media discovery request. Return ONLY a JSON array of objects with fields index (integer), confidence (0-1), reason (short). %s Request: kind=%q title=%q creator=%q year=%q isbn=%q. Candidates: %s", instruction, req.Kind, req.Title, req.Creator, req.Year, req.ISBN, data)
	model, err := s.ollamaModel(ctx)
	if err != nil {
		return nil, err
	}
	body, _ := json.Marshal(map[string]any{"model": model, "stream": false, "format": "json", "messages": []map[string]string{{"role": "user", "content": prompt}}})
	httpReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.OllamaURL+"/api/chat", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("ollama returned %s", resp.Status)
	}
	var decoded ollamaResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&decoded); err != nil {
		return nil, err
	}
	var ranks []struct {
		Index      int     `json:"index"`
		Confidence float64 `json:"confidence"`
		Reason     string  `json:"reason"`
	}
	if err := json.Unmarshal([]byte(decoded.Message.Content), &ranks); err != nil {
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

func (s *server) ollamaModel(ctx context.Context) (string, error) {
	if s.cfg.OllamaModel != "" {
		return s.cfg.OllamaModel, nil
	}
	for _, endpoint := range []string{"/api/ps", "/api/tags"} {
		var models ollamaModelsResponse
		if err := s.ollamaJSON(ctx, endpoint, &models); err != nil || len(models.Models) == 0 {
			continue
		}
		selected := models.Models[0]
		for _, model := range models.Models[1:] {
			if model.Size > selected.Size || (model.Size == selected.Size && model.Name < selected.Name) {
				selected = model
			}
		}
		if selected.Name != "" {
			return selected.Name, nil
		}
	}
	return "", errors.New("Ollama has no available models; set OLLAMA_MODEL or load a model")
}

func (s *server) ollamaJSON(ctx context.Context, endpoint string, output any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.cfg.OllamaURL+endpoint, nil)
	if err != nil {
		return err
	}
	resp, err := s.client.Do(req)
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
	if r.Header.Get("Content-Type") != "application/json" {
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
