"""Deterministic bilingual command grammar (Turkish + English).

No ML. A verb dictionary + a small set of hand-written patterns produces a
ranked list of candidate Intents. This is intentionally narrow: when a
command falls outside the grammar, the parser returns a low-confidence
`unknown` Intent rather than hallucinating. The executor then hands unknown
intents to self-play / the action transformer.

Design rules:
- Every verb has a canonical English form and a set of bilingual aliases.
- An Intent has four slots: verb, obj (what), scope (where/over what), args.
- Ambiguous input returns the top-k candidates, not one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .intent import Intent


@dataclass(frozen=True)
class VerbSpec:
    canonical: str
    aliases: tuple[str, ...]
    default_obj: str = ""


VERBS: tuple[VerbSpec, ...] = (
    VerbSpec("research", ("research", "araştır", "arastir", "search", "ara")),
    VerbSpec("download", ("download", "indir", "fetch", "get", "çek", "cek")),
    VerbSpec("summarize", ("summarize", "summarise", "özetle", "ozetle", "özet", "ozet")),
    VerbSpec("write_code", ("write code for", "code", "kod yaz", "yaz kod", "implement")),
    VerbSpec("make_image", ("make image of", "generate image", "görsel üret", "gorsel uret", "çiz", "ciz")),
    VerbSpec("run", ("run", "execute", "çalıştır", "calistir", "start", "başlat", "baslat")),
    VerbSpec("watch", ("watch", "monitor", "izle", "takip et", "follow")),
    VerbSpec("delete", ("delete", "remove", "sil", "kaldır", "kaldir", "rm")),
    VerbSpec("move", ("move", "taşı", "tasi", "mv")),
    VerbSpec("copy", ("copy", "kopyala", "cp")),
    VerbSpec("rename", ("rename", "yeniden adlandır", "ad değiştir", "ad degistir")),
    VerbSpec("build", ("build", "derle", "compile")),
    VerbSpec("test", ("test", "test et", "run tests")),
    VerbSpec("share", ("share", "paylaş", "paylas", "post", "gönder", "gonder")),
    VerbSpec("organize", ("organize", "düzenle", "duzenle", "tidy", "sort", "sırala", "sirala")),
    VerbSpec("lint", ("lint", "check", "kontrol et")),
    VerbSpec("list", ("list", "ls", "listele", "show", "göster", "goster")),
    VerbSpec("find", ("find", "locate", "bul", "search for")),
    VerbSpec("count", ("count", "say", "sayısını bul", "sayisini bul")),
    VerbSpec("report", ("report", "rapor", "rapor çıkar", "rapor cikar")),
    VerbSpec("clean", ("clean", "temizle", "clear", "purge")),
    VerbSpec("index", ("index", "indeksle")),
    VerbSpec("extract", ("extract", "çıkar", "cikar", "unpack")),
    VerbSpec("convert", ("convert", "dönüştür", "donustur")),
    VerbSpec("inspect", ("inspect", "incele", "bak", "examine")),
    VerbSpec("git_status", ("git status", "git durumu", "repo status", "repo durumu")),
    VerbSpec("commit", ("commit", "git commit", "kaydet repoya")),
    VerbSpec("help", ("help", "yardım", "yardim", "?")),
)


_ALIAS_TO_CANON: dict[str, str] = {}
for spec in VERBS:
    for alias in spec.aliases:
        _ALIAS_TO_CANON[alias.lower()] = spec.canonical


OBJECT_PATTERNS: dict[str, tuple[str, ...]] = {
    "pdf": ("pdf", "pdfs", "pdf'ler", "pdf'leri"),
    "image": ("image", "images", "photo", "photos", "resim", "resimler", "görsel", "gorsel", "png", "jpg", "jpeg"),
    "video": ("video", "videos", "mp4", "mov", "mkv"),
    "repo": ("repo", "repository", "depo"),
    "url": ("url", "link", "address", "adres"),
    "code": ("code", "source", "kod", "kaynak"),
    "note": ("note", "notes", "not", "notlar"),
    "report": ("report", "rapor"),
    "paper": ("paper", "papers", "makale", "makaleler", "article", "articles"),
    "process": ("process", "süreç", "surec"),
    "disk": ("disk", "storage", "depolama"),
    "test": ("test", "tests", "test set"),
    "folder": ("folder", "directory", "klasör", "klasor", "dir", "dizin"),
    "file": ("file", "files", "dosya", "dosyalar"),
}


SCOPE_PATTERNS: dict[str, tuple[str, ...]] = {
    "this_folder": (
        "bu klasör",
        "bu klasordeki",
        "this folder",
        "this directory",
        "şu anki klasör",
        "current folder",
        "current dir",
        "buradaki",
    ),
    "downloads": ("downloads", "indirilenler", "downloads folder", "indirilenler klasörü"),
    "home": ("home", "~", "ev dizini", "home folder"),
}


URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
PATH_RE = re.compile(r"(?:^|\s)((?:\./|/|~/)[^\s'\"]+)")
QUOTED_RE = re.compile(r"[\"“”]([^\"“”]+)[\"“”]|'([^']+)'")


@dataclass(frozen=True)
class Parser:
    verbs: tuple[VerbSpec, ...] = field(default_factory=lambda: VERBS)
    top_k: int = 3

    def parse(self, raw: str) -> list[Intent]:
        text = raw.strip()
        if not text:
            return [Intent(verb="unknown", raw=raw, confidence=0.0)]

        low = text.lower()
        candidates: list[Intent] = []

        verb_hits = self._match_verbs(low)
        if not verb_hits:
            return [Intent(verb="unknown", raw=raw, confidence=0.0)]

        obj = _first_match(OBJECT_PATTERNS, low)
        scope = _first_match(SCOPE_PATTERNS, low)
        args = self._extract_args(text)

        for canonical, conf in verb_hits[: self.top_k]:
            candidates.append(
                Intent(
                    verb=canonical,
                    obj=obj,
                    scope=scope,
                    args=args,
                    raw=raw,
                    confidence=conf,
                )
            )
        return candidates

    def _match_verbs(self, low: str) -> list[tuple[str, float]]:
        hits: dict[str, float] = {}
        for alias, canonical in _ALIAS_TO_CANON.items():
            pattern = rf"(?:^|\W){re.escape(alias)}(?:$|\W)"
            if re.search(pattern, low):
                score = len(alias) / max(len(low), 1)
                score = min(1.0, 0.3 + score)
                prev = hits.get(canonical, 0.0)
                hits[canonical] = max(prev, score)
        return sorted(hits.items(), key=lambda kv: kv[1], reverse=True)

    def _extract_args(self, text: str) -> tuple[tuple[str, str], ...]:
        args: list[tuple[str, str]] = []
        url_match = URL_RE.search(text)
        if url_match:
            args.append(("url", url_match.group(0)))

        for quoted in QUOTED_RE.finditer(text):
            q = quoted.group(1) or quoted.group(2)
            if q:
                args.append(("query", q))
                break

        path_match = PATH_RE.search(" " + text)
        if path_match:
            args.append(("path", path_match.group(1)))

        nums = re.findall(r"\b\d{1,4}\b", text)
        if nums:
            args.append(("n", nums[0]))

        return tuple(args)


def parse(raw: str) -> list[Intent]:
    return Parser().parse(raw)


def _first_match(table: dict[str, tuple[str, ...]], low: str) -> str:
    for key, patterns in table.items():
        for p in patterns:
            if p in low:
                return key
    return ""


def list_verbs() -> list[str]:
    return [v.canonical for v in VERBS]


def verb_aliases() -> Iterable[tuple[str, tuple[str, ...]]]:
    for v in VERBS:
        yield v.canonical, v.aliases
