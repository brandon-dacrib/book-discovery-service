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
	"sort"
	"strings"
	"sync"
	"time"
)

type config struct {
	ListenAddr string
	SearxURL   string
	OllamaURL  string
	OllamaModel string
	Timeout    time.Duration
}

func env(key, fallback string) string { if value := os.Getenv(key); value != "" { return value }; return fallback }

func loadConfig() config {
	return config{
		ListenAddr: env("LISTEN_ADDR", ":8080"),
		SearxURL: strings.TrimRight(env("SEARXNG_URL", ""), "/"),
		OllamaURL: strings.TrimRight(env("OLLAMA_URL", ""), "/"),
		OllamaModel: env("OLLAMA_MODEL", "qwen3:8b"),
		Timeout: 45 * time.Second,
	}
}

type discoverRequest struct {
	Kind string `json:"kind"`
	Title string `json:"title"`
	Creator string `json:"creator"`
	Year string `json:"year,omitempty"`
	ISBN string `json:"isbn,omitempty"`
	Query string `json:"query,omitempty"`
}

type candidate struct {
	Title string `json:"title"`
	URL string `json:"url"`
	Content string `json:"content,omitempty"`
	Engine string `json:"engine,omitempty"`
	Kind string `json:"kind,omitempty"`
	Score float64 `json:"score,omitempty"`
	Confidence float64 `json:"confidence,omitempty"`
	Reason string `json:"reason,omitempty"`
}

type searxResponse struct { Results []struct {
	Title string `json:"title"`
	URL string `json:"url"`
	Content string `json:"content"`
	Engine string `json:"engine"`
	Category string `json:"category"`
	Score float64 `json:"score"`
} `json:"results"` }

type ollamaResponse struct { Message struct { Content string `json:"content"` } `json:"message"` }

type server struct { cfg config; client *http.Client }

func main() {
	cfg := loadConfig()
	if cfg.SearxURL == "" || cfg.OllamaURL == "" { log.Fatal("SEARXNG_URL and OLLAMA_URL are required") }
	s := &server{cfg: cfg, client: &http.Client{Timeout: cfg.Timeout}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("POST /v1/discover", s.discover)
	log.Printf("book-discovery listening on %s", cfg.ListenAddr)
	log.Fatal(http.ListenAndServe(cfg.ListenAddr, securityHeaders(mux)))
}

func securityHeaders(next http.Handler) http.Handler { return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Header().Set("X-Content-Type-Options", "nosniff"); w.Header().Set("Cache-Control", "no-store"); next.ServeHTTP(w, r) }) }
func (s *server) health(w http.ResponseWriter, _ *http.Request) { jsonResponse(w, http.StatusOK, map[string]string{"status":"ok"}) }

func (s *server) discover(w http.ResponseWriter, r *http.Request) {
	var req discoverRequest
	if err := decodeJSON(r, &req); err != nil { http.Error(w, err.Error(), http.StatusBadRequest); return }
	if strings.TrimSpace(req.Title) == "" && strings.TrimSpace(req.Query) == "" { http.Error(w, "title or query is required", http.StatusBadRequest); return }
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout); defer cancel()
	results, err := s.search(ctx, req)
	if err != nil { http.Error(w, err.Error(), http.StatusBadGateway); return }
	ranked, rankingErr := s.rank(ctx, req, results)
	if rankingErr != nil { log.Printf("ollama ranking unavailable: %v", rankingErr); ranked = fallbackRank(req, results) }
	jsonResponse(w, http.StatusOK, map[string]any{"request": req, "results": ranked, "ranked_by": map[bool]string{true:"ollama", false:"deterministic"}[rankingErr == nil]})
}

func (s *server) search(ctx context.Context, req discoverRequest) ([]candidate, error) {
	queries := queryVariants(req)
	var wg sync.WaitGroup; var mu sync.Mutex; all := make([]candidate, 0)
	for _, query := range queries { query := query; wg.Add(1); go func() { defer wg.Done(); found, err := s.searx(ctx, query, req.Kind); if err == nil { mu.Lock(); all = append(all, found...); mu.Unlock() } }() }
	wg.Wait()
	if len(all) == 0 { return nil, errors.New("searxng returned no candidates") }
	return dedupe(all), nil
}

func queryVariants(req discoverRequest) []string {
	base := strings.TrimSpace(req.Query); if base == "" { base = strings.TrimSpace(strings.Join([]string{req.Title, req.Creator}, " ")) }
	kind := strings.ToLower(strings.TrimSpace(req.Kind)); suffix := ""
	switch kind { case "book", "audiobook": suffix = " audiobook"; case "movie": suffix = " movie"; case "tv", "show", "series": suffix = " tv series" }
	queries := []string{base + suffix, fmt.Sprintf("%q %s", req.Title, req.Creator) + suffix}
	if req.ISBN != "" { queries = append(queries, req.ISBN) }
	return uniqueNonEmpty(queries)
}

func (s *server) searx(ctx context.Context, query, kind string) ([]candidate, error) {
	u, _ := url.Parse(s.cfg.SearxURL + "/search"); q := u.Query(); q.Set("q", query); q.Set("format", "json"); q.Set("safesearch", "0"); u.RawQuery = q.Encode()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	resp, err := s.client.Do(req); if err != nil { return nil, err }; defer resp.Body.Close()
	if resp.StatusCode/100 != 2 { return nil, fmt.Errorf("searxng returned %s", resp.Status) }
	var decoded searxResponse; if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&decoded); err != nil { return nil, err }
	out := make([]candidate, 0, len(decoded.Results)); for _, item := range decoded.Results { if item.URL == "" || item.Title == "" { continue }; out = append(out, candidate{Title:item.Title, URL:item.URL, Content:item.Content, Engine:item.Engine, Kind:kind, Score:item.Score}) }; return out, nil
}

func (s *server) rank(ctx context.Context, req discoverRequest, candidates []candidate) ([]candidate, error) {
	if len(candidates) > 40 { candidates = candidates[:40] }
	data, _ := json.Marshal(candidates)
	prompt := fmt.Sprintf("You rank search candidates for a media discovery request. Return ONLY a JSON array of objects with fields index (integer), confidence (0-1), reason (short). Prefer exact title and creator, correct media type, and authoritative pages. Request: kind=%q title=%q creator=%q year=%q isbn=%q. Candidates: %s", req.Kind, req.Title, req.Creator, req.Year, req.ISBN, data)
	body, _ := json.Marshal(map[string]any{"model":s.cfg.OllamaModel,"stream":false,"format":"json","messages":[]map[string]string{{"role":"user","content":prompt}}})
	httpReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.OllamaURL+"/api/chat", bytes.NewReader(body)); httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq); if err != nil { return nil, err }; defer resp.Body.Close(); if resp.StatusCode/100 != 2 { return nil, fmt.Errorf("ollama returned %s", resp.Status) }
	var decoded ollamaResponse; if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&decoded); err != nil { return nil, err }
	var ranks []struct { Index int `json:"index"`; Confidence float64 `json:"confidence"`; Reason string `json:"reason"` }; if err := json.Unmarshal([]byte(decoded.Message.Content), &ranks); err != nil { return nil, err }
	out := make([]candidate, 0, len(ranks)); for _, rank := range ranks { if rank.Index < 0 || rank.Index >= len(candidates) { continue }; item := candidates[rank.Index]; item.Confidence = clamp(rank.Confidence); item.Reason = rank.Reason; out = append(out, item) }; sort.SliceStable(out, func(i,j int) bool { return out[i].Confidence > out[j].Confidence }); return out, nil
}

func fallbackRank(req discoverRequest, in []candidate) []candidate { want := strings.ToLower(strings.Join([]string{req.Title, req.Creator}, " ")); out := append([]candidate(nil), in...); for i := range out { hay := strings.ToLower(out[i].Title+" "+out[i].Content); if strings.Contains(hay, strings.ToLower(req.Title)) { out[i].Confidence += .55 }; if req.Creator != "" && strings.Contains(hay, strings.ToLower(req.Creator)) { out[i].Confidence += .35 }; out[i].Confidence = clamp(out[i].Confidence); out[i].Reason = "title/creator text match" }; sort.SliceStable(out, func(i,j int) bool { return out[i].Confidence > out[j].Confidence }); _ = want; return out }
func dedupe(in []candidate) []candidate { seen := map[string]bool{}; out := make([]candidate,0,len(in)); for _, item := range in { key := strings.ToLower(item.URL); if !seen[key] { seen[key]=true; out=append(out,item) } }; return out }
func uniqueNonEmpty(in []string) []string { seen:=map[string]bool{}; out:=[]string{}; for _, v:=range in { v=strings.TrimSpace(v); if v!=""&&!seen[v] {seen[v]=true;out=append(out,v)} }; return out }
func clamp(v float64) float64 { if v<0{return 0}; if v>1{return 1}; return v }
func decodeJSON(r *http.Request, dst any) error { if r.Header.Get("Content-Type") != "application/json" { return errors.New("Content-Type must be application/json") }; dec:=json.NewDecoder(io.LimitReader(r.Body, 1<<20)); dec.DisallowUnknownFields(); return dec.Decode(dst) }
func jsonResponse(w http.ResponseWriter, status int, value any) { w.Header().Set("Content-Type","application/json"); w.WriteHeader(status); _=json.NewEncoder(w).Encode(value) }
