# Book Discovery Service

Resolves a book or audiobook to the **correct** Shelfarr work, and provides a
web-search discovery API alongside it. It does not download files.

## Why this exists

Shelfarr's metadata search returns the real book alongside study guides,
lesson plans, academic theses, box sets, and other volumes in the same series —
and reports the **same confidence score for every row**, so neither its
ordering nor its score can separate them. Requesting the first result therefore
files the wrong work a large fraction of the time.

Measured on a live instance over 75 titles across 25 genres, from literary
fiction and translated works to cookbooks, poetry, graphic novels, and
self-published progression fantasy:

| selection strategy | correct |
| --- | --- |
| Shelfarr's first result | 20/75 |
| **this service** | **75/75** |

Validated on further sets, all with no model configured:

| suite | result |
| --- | --- |
| held-out set, 49 books scored as both ebook and audiobook | 98/98 |
| messy input, live — subtitles, series prefixes, typos, missing punctuation | 12/12 |
| adversarial, live — invented titles and wrong authors | 6/6 |

The held-out titles share nothing with the set used for tuning, and format
filtering behaves identically for ebooks and audiobooks.

## Refusing is a feature

Hardcover fuzzy-matches, so a query for a book that does not exist still comes
back full of real ones: `Qwertyuiop Asdfghjkl` returned *Back in Time with
Thomas Edison*. Filing that would acquire a random book, so resolution requires
an actual title match and otherwise fails with `422`.

A wrong author is treated as a misremembered one when the title is unique —
`Project Hail Mary` by "Frank Herbert" still resolves to Andy Weir's book — but
refuses when several real works share the title, because guessing between them
is how the wrong book gets acquired.

## On additional metadata sources

Resolution is only as good as the catalogue behind it, so a second source was
investigated. Shelfarr's `/api/v1/search` returns `hardcover` results
exclusively, including for queries Hardcover answers poorly and for nonsense
queries, so a Google Books key configured in Shelfarr does not reach this
endpoint. Google Books' own API refuses keyless traffic once the shared
anonymous quota is spent, and Open Library — which is keyless and does work —
disagrees with Hardcover on publication years often enough to be unsafe as a
matching signal.

None of that is a blocker today: a request needs a Shelfarr `work_id`, which
only Shelfarr can mint, and the deterministic matcher already resolves every
title in the suites above. A second source would help only where the catalogue
lacks the work entirely, which it cannot fix either.

## No language model is required

Resolution was benchmarked with and without one, replaying an identical cached
candidate set through the real service so the arms differ only in the model:

| ranking model | size | correct | wall clock |
| --- | --- | --- | --- |
| **none — deterministic only** | **0** | **75/75** | **~0s** |
| qwen2.5:3b | 1.9 GB | 75/75 | 0.4 min |
| gemma3:27b | 17.4 GB | 75/75 | 3.3 min |
| qwen2.5:0.5b | 0.4 GB | 73/75 | 0.2 min |

A model changes no outcome, and the smallest one is actively worse — it chose
a Turkish edition of *Verity* that the deterministic path rejects. Embedding a
model in the image was considered and **rejected on this evidence**: it would
add 0.4–2 GB to a 7 MB image and buy nothing.

`OLLAMA_URL` is therefore optional. Leave it unset for resolution. It still
drives `/v1/discover` and `/v1/recommendations`, which rank open web results
and do need judgement.

The failures were not obscure. *Kafka on the Shore* resolved to a Gale Cengage
study guide, *The Sandman* to a BookRags lesson plan, *Leviathan Wakes* to a
box set, *Dungeon Crawler Carl* to book 8 of the series, and *Circe* to a blank
notebook sold under the novel's name.

What was wrong, in descending order of impact:

1. **Recall, not ranking, was the binding constraint.** At Shelfarr's default
   result count neither *Dune* nor *Atomic Habits* appeared at all — summaries
   and companions filled every slot. Asking for more candidates fixed both.
2. **The obvious query is not always the best one.** Adding the author usually
   sharpens the match but sometimes collapses it: `Charlotte's Web E. B. White`
   returned three results and none was the book, while `Charlotte's Web`
   returned fifteen and did. A title-only retry runs when the first query comes
   back without anything titled like the request.
3. **The work is deterministic.** Filtering to candidates whose title and
   author actually match the request removes omnibus editions, workbooks,
   merchandise, comic adaptations, translations, and other volumes in a series
   by construction — which is why no model is needed.
4. **Some titles are only reachable by asking differently.** A series with a
   comic adaptation buries the novel: `The Eye of the World` and
   `The Eye of the World Robert Jordan` return twenty numbered comic issues and
   nothing else, while adding `novel` surfaces the book. That escalation runs
   only when the earlier queries came back without the work.
5. **Refuse rather than guess.** When every candidate is a study guide,
   adaptation, or piece of merchandise, resolution fails instead of filing the
   closest thing, because acquiring the wrong book is worse than acquiring
   nothing. A candidate titled exactly as requested is exempt, so a real book
   called *The Journal of Best Practices* still resolves.

Two limits are the catalogue's, not this service's. It carries no English
edition of *Circe*, only the French *Circé*, so that is what resolves; and
where a work is absent entirely, no amount of ranking invents it.

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
# Optional. Only needed for /v1/discover and /v1/recommendations; work
# resolution is deterministic and does not use a model. See the benchmark above.
OLLAMA_URL=
# Optional. If omitted, a model is chosen automatically; see Model selection.
OLLAMA_MODEL=
# Include any sub-path your Shelfarr is mounted under. Rails deployments that
# set RAILS_RELATIVE_URL_ROOT serve the API below that prefix, so the value
# must be e.g. http://shelfarr:5056/requests, not just http://shelfarr:5056.
SHELFARR_URL=http://shelfarr:5056
SHELFARR_API_TOKEN=...
# Shelfarr attributes requests to a user; creates fail without it.
SHELFARR_USER_ID=1
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
