"""Source parsing into canonical records.

Parsers are registered by file suffix and return one or more records. Adding a
format means registering a callable; nothing downstream changes.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path

from .records import Record, sha256_bytes

Parser = Callable[[Path, bytes], Iterator[Record]]
_REGISTRY: dict[str, Parser] = {}


def register(suffix: str) -> Callable[[Parser], Parser]:
    def deco(fn: Parser) -> Parser:
        _REGISTRY[suffix.lower()] = fn
        return fn
    return deco


def supported_suffixes() -> list[str]:
    return sorted(_REGISTRY)


@register(".txt")
def parse_text(path: Path, data: bytes) -> Iterator[Record]:
    text = data.decode("utf-8", errors="replace").strip()
    if text:
        yield Record(record_id=f"{path.stem}#0", text=text,
                     source_id=str(path.name), source_hash=sha256_bytes(data),
                     metadata={"format": "txt"})


@register(".md")
def parse_markdown(path: Path, data: bytes) -> Iterator[Record]:
    """Split on ATX headings so each section becomes a retrievable unit."""
    text = data.decode("utf-8", errors="replace")
    parts = re.split(r"^(#{1,6}\s+.*)$", text, flags=re.MULTILINE)
    chunks: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for part in parts:
        if re.match(r"^#{1,6}\s+", part or ""):
            if buf and "".join(buf).strip():
                chunks.append((heading, "".join(buf).strip()))
            heading = part.strip("# ").strip()
            buf = []
        else:
            buf.append(part or "")
    if buf and "".join(buf).strip():
        chunks.append((heading, "".join(buf).strip()))

    digest = sha256_bytes(data)
    for i, (head, body) in enumerate(chunks):
        yield Record(record_id=f"{path.stem}#{i}",
                     text=f"{head}\n{body}".strip() if head else body,
                     source_id=str(path.name), source_hash=digest,
                     metadata={"format": "md", "heading": head, "chunk": i})


@register(".jsonl")
def parse_jsonl(path: Path, data: bytes) -> Iterator[Record]:
    digest = sha256_bytes(data)
    for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        text = obj.get("text") or obj.get("content") or ""
        if not text:
            continue
        yield Record(record_id=f"{path.stem}#{i}", text=str(text).strip(),
                     source_id=str(path.name), source_hash=digest,
                     metadata={"format": "jsonl",
                               **{k: v for k, v in obj.items()
                                  if k not in ("text", "content")}})


def parse_file(path: Path) -> list[Record]:
    parser = _REGISTRY.get(path.suffix.lower())
    if parser is None:
        return []
    data = path.read_bytes()
    out = []
    for rec in parser(path, data):
        rec.applied("parse", "1.0", parser=path.suffix.lower(),
                    source_bytes=len(data))
        out.append(rec)
    return out


def ingest(root: Path) -> list[Record]:
    """Parse every supported file under a directory, in stable order."""
    records: list[Record] = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            records.extend(parse_file(p))
    return records


__all__ = ["Parser", "ingest", "parse_file", "register", "supported_suffixes"]
