#!/bin/sh
# Clone rag-foundation read-only at the pinned commit, for the
# question file Half 2 draws from. The evaluated code itself is
# vendored in src/ragref/ (see its PROVENANCE.md).
set -eu
PIN=007e33238e6ce49100650b30f6820149410f022b
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REF="$ROOT/data/ref/rag-foundation"
if [ ! -d "$REF/.git" ]; then
  mkdir -p "$ROOT/data/ref"
  git clone -q https://github.com/JesunAhmadUshno/rag-foundation "$REF"
fi
git -C "$REF" checkout -q "$PIN"
echo "rag-foundation pinned at $PIN in data/ref/rag-foundation"
