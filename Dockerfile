# syntax=docker/dockerfile:1

# ── Builder ────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

ARG GO_VERSION=1.23.4

# curl + ca-certificates for Go download. build-essential provides g++ for nvcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz \
    | tar -C /usr/local -xz

ENV PATH="/usr/local/go/bin:${PATH}"
ENV CGO_ENABLED=1
ENV GOPROXY=https://proxy.golang.org,direct

WORKDIR /src

# Separate layer for module downloads so source changes don't bust the cache.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Compile the CUDA shared library.
# sm_90 targets H100 (Hopper). For A100 use sm_80; for B200 use sm_100.
# Add multiple -gencode flags if you need a fat binary across architectures.
RUN nvcc -O2 -arch=sm_90 \
        -shared -Xcompiler -fPIC \
        cuda/gpu_pulse.cu \
        -o cuda/libgpupulse.so \
        -lcudart

# Compile the Go agent. LD_LIBRARY_PATH lets cgo resolve the .so at link time.
# The binary embeds rpath=/usr/local/lib where the runtime stage places the .so.
RUN CGO_CFLAGS="-I/src/cuda" \
    CGO_LDFLAGS="-L/src/cuda -lgpupulse -lcudart -lstdc++ -Wl,-rpath,/usr/local/lib" \
    LD_LIBRARY_PATH=/src/cuda \
    go build \
        -tags cuda \
        -ldflags="-s -w" \
        -o /straggler-shield \
        ./cmd/agent

# ── Runtime ────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-base-ubuntu22.04

# libstdc++6 is required by libgpupulse.so at runtime.
# libcudart is already provided by the base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/cuda/libgpupulse.so /usr/local/lib/libgpupulse.so
COPY --from=builder /straggler-shield          /usr/local/bin/straggler-shield

# Refresh the dynamic linker cache. The binary's embedded rpath also points to
# /usr/local/lib, so either path resolves the library — ldconfig is belt-and-
# suspenders for any ld.so.conf-based lookup that ignores rpath.
RUN ldconfig

EXPOSE 9090

ENTRYPOINT ["/usr/local/bin/straggler-shield"]
