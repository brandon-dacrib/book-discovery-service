# Book Discovery Service

An internal, read-only discovery API for books, audiobooks, movies, and TV.
It fans out query variants to SearXNG, deduplicates candidates, and asks
Ollama to rank them by title, creator, year, ISBN, and media type. It does not
download files or automatically select an acquisition.

## Configuration

All endpoints are environment variables; no service URLs or credentials are
stored in the repository.

```text
SEARXNG_URL=https://search.example.invalid
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=qwen3:8b
LISTEN_ADDR=:8080
```

## API

```sh
curl -X POST http://localhost:8080/v1/discover \
  -H 'content-type: application/json' \
  -d '{"kind":"audiobook","title":"Cappadonna","creator":"Jahquel J."}'
```

The response includes ranked candidate URLs and a confidence score. If Ollama
is unavailable, the service returns deterministic title/creator text matching
with `ranked_by: deterministic`.

## Run

```sh
docker build -t book-discovery-service .
docker run --rm -p 8080:8080 \
  -e SEARXNG_URL=https://search.example.invalid \
  -e OLLAMA_URL=http://192.0.2.10:11434 \
  book-discovery-service
```
