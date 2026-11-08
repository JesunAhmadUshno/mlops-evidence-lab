#!/bin/sh
# Fetch the credit-risk data file from the pinned credit-risk-lab
# commit, read-only, and verify its SHA-256 against the value the
# subject repo published in its own run manifest.
set -eu
PIN=fee580d0beee43c76dc57c6c366c211533054956
EXPECT=30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REF="$ROOT/data/ref/credit-risk-lab"
if [ ! -d "$REF/.git" ]; then
  mkdir -p "$ROOT/data/ref"
  git clone -q https://github.com/JesunAhmadUshno/credit-risk-lab "$REF"
fi
git -C "$REF" checkout -q "$PIN"
SRC="$REF/data/default of credit card clients.xls"
GOT=$(sha256sum "$SRC" | awk '{print $1}')
if [ "$GOT" != "$EXPECT" ]; then
  echo "SHA-256 mismatch: got $GOT, expected $EXPECT" >&2
  exit 1
fi
cp "$SRC" "$ROOT/data/credit_card.xls"
echo "data/credit_card.xls verified: sha256 $GOT"
