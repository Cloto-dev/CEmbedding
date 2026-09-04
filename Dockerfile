# CEmbedding as a container.
#
# What this image serves: the REST surface -- /embed, and the index endpoints
# (/index, /search, /remove, /purge) while the index is enabled -- on 8401,
# bound to every interface, with the model and the index on a volume.
#
# EMBEDDING_HTTP_HOST is the one value this image must override. The server
# defaults to 127.0.0.1, which inside a container is the container's own
# loopback: a published port would forward to a socket nothing is listening on,
# and the symptom is a refused connection that reads like a crash. Binding every
# interface is not a decision about who may call -- a tunnel or a reverse proxy
# reaches loopback just as well, which is why the server refuses to treat the
# bind address as a boundary. Set CEMBEDDING_AUTH_TOKEN before publishing this
# port anywhere a stranger can reach.
#
# The weights are not in the image. An ONNX provider fetches what it needs on
# first use, so a container started against an empty volume reaches Hugging Face
# and cannot answer until that download finishes -- half a gigabyte for bge-m3,
# with no progress visible to whoever is waiting on the port. Fill the volume
# once, before anything depends on the service:
#
#   docker run --rm -v cembedding-data:/data <image> \
#       cembedding-download-model --model jina-v5-nano
#
# ONNX_MODEL_DIR below points both that command and the server at the same
# directory, so the model a run downloads is the model the next run loads. One
# model per volume: the variable names a directory, not a collection.
#
# Baking the weights in instead would trade a small image for a gigabyte-scale
# one and freeze a choice the environment is supposed to make.

FROM python:3.12-slim AS build

WORKDIR /src

# Only the wheel's inputs; .dockerignore admits nothing else. Copied as separate
# layers from the install so editing the package does not re-resolve the build
# backend.
COPY pyproject.toml README.md LICENSE ./
COPY cembedding/ ./cembedding/

# The wheel, not the source tree. The runtime stage installs this artifact, so
# the container runs what a `pip install cembedding` user runs rather than a
# second copy of the repository that happens to sit on sys.path -- the two
# diverge exactly when the packaging is wrong, which is the case worth catching.
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .


# 3.12 rather than the newest release: it is one of the two versions CI runs the
# suite on, and onnxruntime publishes wheels for it on every platform this image
# is built for. A base image nothing tests is a dependency resolution waiting to
# happen at `docker build` time.
FROM python:3.12-slim

# A fixed uid, because it outlives the image. A named volume inherits ownership
# from the image and needs nothing; a bind-mounted host directory does not, so
# the host side has to be writable by this uid (or `--user "$(id -u)"` passed).
# Naming the number here is what makes that instruction possible to follow.
RUN useradd --system --create-home --uid 10001 cembedding

# The onnx extra, because on-device inference is the reason to run this next to
# the service that calls it; the extra is named rather than its contents listed,
# so the image cannot drift from what `pip install "cembedding[onnx]"` means.
COPY --from=build /wheels/*.whl /tmp/
RUN pip install --no-cache-dir "$(echo /tmp/*.whl)[onnx]" && rm -rf /tmp/*.whl

# 8401 is the server's own default REST port; it is repeated here so that
# changing one without the other cannot go unnoticed, and so EXPOSE below names
# a port something actually binds.
ENV EMBEDDING_HTTP_HOST=0.0.0.0 \
    EMBEDDING_HTTP_PORT=8401 \
    EMBEDDING_INDEX_DB_PATH=/data/embedding_index.db \
    ONNX_MODEL_DIR=/data/model

# Created before the volume is declared, so a fresh named volume inherits this
# ownership instead of arriving as root-owned and unwritable.
RUN install -d -o cembedding -g cembedding /data
VOLUME ["/data"]

USER cembedding
WORKDIR /home/cembedding
EXPOSE 8401

# Liveness only, and deliberately not a claim about health: it asks whether the
# port is bound and the application answers HTTP. Any status counts, because an
# authenticated deployment answers 401 to an anonymous probe and a 401 is a
# served response; a GET on a POST-only route answers 405, which is one too.
# What the probe cannot tell you is whether the model finished loading -- that
# is what a first /embed call answers, and it needs a request body this must not
# invent. http.client rather than urllib: it returns the status instead of
# raising on 4xx, so the probe needs no exception handling, while a refused
# connection still raises and exits non-zero.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os,http.client;c=http.client.HTTPConnection('127.0.0.1',int(os.environ.get('EMBEDDING_HTTP_PORT','8401')),timeout=4);c.request('GET','/embed');print(c.getresponse().status)"]

LABEL org.opencontainers.image.title="CEmbedding" \
      org.opencontainers.image.description="Local-first embedding server: /embed plus a persistent vector index (MCP)" \
      org.opencontainers.image.source="https://github.com/Cloto-dev/CEmbedding" \
      org.opencontainers.image.licenses="MIT"

# Why the command is not just `cembedding`: under the default stdio transport
# the MCP loop owns the foreground and the REST endpoint runs behind it, so the
# process shuts down when stdin reaches EOF. A container started without an
# interactive stdin gets EOF immediately -- the endpoint starts, logs that it
# started, and the container exits 0, which reads as a clean run rather than as
# a service that never served. Holding stdin open from a process the shell
# leaves behind keeps the REST surface up, and `exec` keeps the server itself as
# PID 1 so it receives the stop signal directly.
#
# Overriding this command (`docker run <image> cembedding-download-model ...`)
# is the intended way to run the one-shot tools, which want an ordinary exit.
CMD ["bash", "-c", "exec cembedding < <(sleep infinity)"]
