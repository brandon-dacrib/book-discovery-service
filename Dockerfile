# The toolchain version determines which standard library is linked in, and
# most of this binary's attack surface is stdlib net/http. Keep this on a
# current release rather than pinning back to go.mod's minimum, and re-check
# with `govulncheck` after bumping.
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o /out/book-discovery ./cmd/book-discovery

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/book-discovery /book-discovery
USER nonroot:nonroot
EXPOSE 8080
ENTRYPOINT ["/book-discovery"]
