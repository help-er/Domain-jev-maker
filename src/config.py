"""Shared configuration.

`BASE` is the backbone every script loads. Override it without editing code:

    export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct     # or a local path
"""
import os

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Prepended to every prompt. It is deliberately short: in the pointer layout the
# model is never asked to emit a label, so the system message only has to set
# the frame.
SYSTEM = ("You answer decision questions about the given state. "
          "Reply with exactly one of the listed options and nothing else.")

SEED = 20260918
