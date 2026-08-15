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
# Optional. If omitted, /api/ps is checked first, then /api/tags.
OLLAMA_MODEL=
SHELFARR_URL=http://shelfarr:5056
SHELFARR_API_TOKEN=...
# Optional protection for this service's write endpoint.
SERVICE_API_TOKEN=...
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

`POST /v1/recommendations` uses the same pipeline with a related-media intent,
so it can support recommendation screens for books now and movies/TV later:

```sh
curl -X POST http://localhost:8080/v1/recommendations \
  -H 'content-type: application/json' \
  -d '{"kind":"book","title":"The Expanse","creator":"James S. A. Corey","preferences":"space opera, political science fiction"}'
```

## Creating a Shelfarr request

`POST /v1/requests` resolves a title through Shelfarr's metadata API when no
`work_id` is supplied, then creates the request through Shelfarr's scoped API.
It defaults to an audiobook unless `kind` is `ebook` or `book`.

```sh
curl -X POST http://localhost:8080/v1/requests \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $SERVICE_API_TOKEN" \
  -d '{"kind":"audiobook","title":"Cappadonna","creator":"Jahquel J.","notes":"Created from discovery"}'
```

## Run

```sh
docker build -t book-discovery-service .
docker run --rm -p 8080:8080 \
  -e SEARXNG_URL=https://search.example.invalid \
  -e OLLAMA_URL=http://192.0.2.10:11434 \
  book-discovery-service
```
