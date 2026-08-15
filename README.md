# Book Discovery Service

Resolves a book or audiobook to the **correct** Shelfarr work, and provides a
web-search discovery API alongside it. It does not download files.

## Why this exists

Shelfarr's metadata search returns the real book alongside study guides,
lesson plans, academic theses, box sets, and other volumes in the same series —
and reports the **same confidence score for every row**, so neither its
ordering nor its score can separate them. Requesting the first result therefore
files the wrong work a large fraction of the time.

Measured on a live instance across 25 genres, from literary fiction and
translated works to cookbooks, poetry, graphic novels, and self-published
progression fantasy:

| selection strategy | correct |
| --- | --- |
| Shelfarr's first result | 15/50 |
| **this service** | **50/50** |

A third set of 25 could not be scored cleanly: Shelfarr's upstream metadata
provider rate limits, and returns HTTP 200 with an empty result set when it
does, so a third of that run failed for reasons unrelated to selection. See
the note on retries below.

The failures were not obscure. *Kafka on the Shore* resolved to a Gale Cengage
study guide, *The Sandman* to a BookRags lesson plan, *Leviathan Wakes* to a
box set, *Dungeon Crawler Carl* to book 8 of the series, and *Circe* to a blank
notebook sold under the novel's name.

Three things were wrong, and the order matters:

1. **Recall, not ranking, was the binding constraint.** At Shelfarr's default
   result count neither *Dune* nor *Atomic Habits* appeared at all — summaries
   and companions filled every slot. Asking for more candidates fixed both.
2. **The obvious query is not always the best one.** Adding the author usually
   sharpens the match but sometimes collapses it: `Charlotte's Web E. B. White`
   returned three results and none was the book, while `Charlotte's Web`
   returned fifteen and did. A title-only retry runs when the first query comes
   back without anything titled like the request.
3. **Most of the remaining work is deterministic.** Filtering to candidates
   whose title and author actually match the request removes omnibus editions,
   workbooks, merchandise, comic adaptations, and other volumes in a series by
   construction. The model is only consulted for what is left. Across the 50
   scored resolutions, **58% never called a model at all** — the filters left a
   single candidate — and the rest took a few seconds each.

Shelfarr's upstream provider signals throttling as HTTP 200 with an empty
result set, which is indistinguishable from "no such book". The service waits
and retries once so a throttle does not get reported as a miss, and issues the
broader second query only when the first did not already contain a title match.

`POST /v1/resolve` performs that selection and returns a `work_id`, so the
result is actionable rather than advisory. `POST /v1/requests` uses the same
path before filing.

The service also fans out query variants to SearXNG and ranks the resulting
pages (`/v1/discover`, `/v1/recommendations`). That path returns URLs rather
than identifiers, so treat it as a research aid, not an acquisition source.

## Configuration

All endpoints are environment variables; no service URLs or credentials are
stored in the repository.

```text
SEARXNG_URL=https://search.example.invalid
OLLAMA_URL=http://ollama:11434
# Optional. If omitted, a model is chosen automatically; see Model selection.
OLLAMA_MODEL=
# Include any sub-path your Shelfarr is mounted under. Rails deployments that
# set RAILS_RELATIVE_URL_ROOT serve the API below that prefix, so the value
# must be e.g. http://shelfarr:5056/requests, not just http://shelfarr:5056.
SHELFARR_URL=http://shelfarr:5056
SHELFARR_API_TOKEN=...
# Optional protection for this service's write endpoint.
SERVICE_API_TOKEN=...
STATE_PATH=/data/discovery-state.json
LISTEN_ADDR=:8080
# Budget for the SearXNG fan-out (Go duration).
REQUEST_TIMEOUT=45s
# Budget for one Ollama ranking call. A cold model load can take minutes, so
# this is deliberately generous.
OLLAMA_TIMEOUT=10m
```

SearXNG must have `json` in `search.formats`; it is not enabled by default.

## Model selection

When `OLLAMA_MODEL` is unset the service picks one itself:

1. `/api/ps` is consulted first, because a model already resident answers far
   faster than one that has to be paged in from disk.
2. `/api/tags` is the fallback when nothing loaded is suitable.
3. Within either list it prefers the **largest** model, since this service backs
   background work where ranking quality matters more than tokens per second.

Models advertising the `thinking` capability are skipped. Their reasoning
preamble is slow and tends to corrupt the JSON payload. Models whose names mark
them as specialized (`coder`, `embed`, …) are skipped too, since judging whether
a search result matches a title is not what they are trained for. Either can
still be forced through `OLLAMA_MODEL`.

Ranking replies are parsed permissively. Local models under `format: json`
variously return a bare array, a single object, an array wrapped under some key,
or JSON preceded by chat-harness tokens — all four are accepted. When a reply
cannot be used at all, the service falls back to deterministic title/creator
matching and says so in `ranked_by`.

## Security notes

- `/v1/requests` and `/v1/history` are gated by `SERVICE_API_TOKEN`, compared in
  constant time. **If that variable is unset both endpoints are open**; the
  service logs a warning at startup. `/v1/discover` and `/v1/recommendations`
  are unauthenticated by design.
- Persisted state is written through a temporary file at mode `0600` and
  renamed into place, so history is not world-readable.
- Search result text is fed to the model as ranking input. The model's replies
  are only ever read as an index, a confidence, and a reason string, and the
  index is bounds-checked against the candidate list, so a malicious page cannot
  redirect a request. The `reason` string is model-authored: display it as
  untrusted text.
- The container's Go toolchain determines which standard library ships in the
  binary. Re-run `govulncheck` after changing the base image.

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

## Resolving a work

`POST /v1/resolve` looks the title up in Shelfarr, picks the work that is
actually being asked for, and returns its `work_id` without filing anything.
Use it to preview what a request would target.

```sh
curl -X POST http://localhost:8080/v1/resolve \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $SERVICE_API_TOKEN" \
  -d '{"kind":"audiobook","title":"Kafka on the Shore","creator":"Haruki Murakami"}'
```

```json
{"work_id":"hardcover:207877","title":"Kafka on the Shore","author":"Haruki Murakami",
 "year":"2001","book_type":"audiobook","candidates":8}
```

Only works Shelfarr lists in the requested format are eligible, so an audiobook
request is never filed against an ebook-only edition. The model chooses among
those candidates by index and cannot invent a work; if it is unreachable, the
deterministic filter still removes the most common derivative editions.

## Creating a Shelfarr request

`POST /v1/requests` resolves a title through Shelfarr's metadata API when no
`work_id` is supplied, then creates the request through Shelfarr's scoped API.
It defaults to an audiobook unless `kind` is `ebook` or `book`.
Send an `Idempotency-Key` header when a caller may retry a request; repeated
keys return the recorded response instead of creating another request.

`GET /v1/capabilities` describes supported media kinds, operations, and
configured request backends for future *arr integrations.

```sh
curl -X POST http://localhost:8080/v1/requests \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $SERVICE_API_TOKEN" \
  -d '{"kind":"audiobook","title":"Cappadonna","creator":"Jahquel J.","notes":"Created from discovery"}'
```

Shelfarr issues its own API key; generate one in its UI and pass it as
`SHELFARR_API_TOKEN`. A wrong or missing key surfaces as `502` from this
service with the upstream `401` in the message.

Discovery, recommendation, and request intents are retained in the JSON state
file named by `STATE_PATH` (up to 500 recent entries). `GET /v1/history` reads
that history and uses `SERVICE_API_TOKEN` when configured.

## Tests

```sh
go test ./...          # unit and handler tests, no network required
go vet ./...
```

The Shelfarr tests run against a stub that mirrors the live contract: bearer
auth, a `401` without it, and the API hanging off whatever base path
`SHELFARR_URL` carries. The ranking-parser tests pin the reply shapes real
local models produce.

## Run

```sh
docker build -t book-discovery-service .
docker run --rm -p 8080:8080 \
  -e SEARXNG_URL=https://search.example.invalid \
  -e OLLAMA_URL=http://192.0.2.10:11434 \
  book-discovery-service
```
