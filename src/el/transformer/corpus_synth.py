"""Synthetic corpus builder for the Action Transformer.

We can't ship NL2Bash inside the wheel (license + size), and the bundled
tldr/man subset is tiny (~30 examples). This module deterministically
generates a few thousand `(intent, actions, reward)` triples by combining:

- the parser's canonical verbs and bilingual aliases (TR + EN);
- a per-verb schema bank that maps each verb to several action templates;
- a placeholder bank (paths, queries, file types, urls) that fills the
  templates;
- a small set of *negative* examples (wrong action for the verb) that
  receive reward 0 so the reward-conditioned model learns what NOT to do.

The output is a list of `SeedExample` instances ready to be tokenized
and fed into `el.transformer.train`. Determinism: the generator takes a
seed; same seed → same corpus.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from ..intent import Intent
from ..primitives import Action
from .dataset import SeedExample


PATH_BANK: tuple[str, ...] = (
    ".", "./src", "./tests", "./docs", "./build", "./dist", "./.cache",
    "/tmp", "/var/log", "/etc", "~/Downloads", "~/Documents", "~/Pictures",
    "~/Music", "~/Videos", "~/Desktop", "~/projects", "~/code", "~/notes",
    "./data", "./assets", "./scripts", "./logs", "./backups",
)

FILE_BANK: tuple[str, ...] = (
    "README.md", "LICENSE", "pyproject.toml", "package.json", "Makefile",
    "config.yaml", "main.py", "app.py", "index.html", "style.css",
    "notes.txt", "report.pdf", "data.csv", "schema.json", "Dockerfile",
    "requirements.txt", ".gitignore", "setup.py", "Cargo.toml",
)

QUERY_BANK: tuple[str, ...] = (
    "TODO", "FIXME", "BUG", "HACK", "XXX",
    "import torch", "def main", "class ", "raise NotImplementedError",
    "password", "api_key", "localhost", "404", "ERROR",
)

URL_BANK: tuple[str, ...] = (
    "https://example.com",
    "https://raw.githubusercontent.com/test/test/main/file.txt",
    "https://api.github.com/repos/python/cpython",
    "https://httpbin.org/get",
    "https://www.gutenberg.org/files/1342/1342-0.txt",
)

EXT_BANK: tuple[str, ...] = ("py", "txt", "md", "json", "csv", "log", "yml", "html")

PROCESS_PATTERNS: tuple[str, ...] = ("python", "node", "nginx", "postgres", "redis", "ssh")


# Per-verb schema bank. Each schema is (raw_template, intent_args_template,
# build_actions(rng, args) -> list[Action]). The raw template is rendered
# with placeholder values; the same values are passed to the action builder
# so command and actions stay in sync.

SchemaFn = Callable[[random.Random, dict[str, str]], list[Action]]


@dataclass(frozen=True)
class Schema:
    raw_templates: tuple[str, ...]   # bilingual variants
    obj: str
    scope: str
    placeholders: tuple[str, ...]    # which keys to fill
    build: SchemaFn


def _path(rng: random.Random) -> str:
    return rng.choice(PATH_BANK)


def _file(rng: random.Random) -> str:
    return rng.choice(FILE_BANK)


def _query(rng: random.Random) -> str:
    return rng.choice(QUERY_BANK)


def _url(rng: random.Random) -> str:
    return rng.choice(URL_BANK)


def _ext(rng: random.Random) -> str:
    return rng.choice(EXT_BANK)


def _process(rng: random.Random) -> str:
    return rng.choice(PROCESS_PATTERNS)


PLACEHOLDER_FNS: dict[str, Callable[[random.Random], str]] = {
    "path": _path,
    "src": _path,
    "dst": _path,
    "file": _file,
    "query": _query,
    "url": _url,
    "ext": _ext,
    "process": _process,
}


# ---- per-verb schemas ----

def _list_schema() -> tuple[Schema, ...]:
    def b1(rng, a): return [Action.make("file_list", path=a["path"])]
    def b2(rng, a): return [Action.make("sh", cmd=f"ls -la {a['path']}", timeout=10)]
    return (
        Schema(("list files in {path}", "list {path}", "ls {path}", "{path} listele",
                "listele {path}", "{path} klasörünü göster", "show {path}"),
               obj="folder", scope="path", placeholders=("path",), build=b1),
        Schema(("list everything in {path}", "{path} her şeyi göster",
                "show all entries in {path}"),
               obj="folder", scope="path", placeholders=("path",), build=b2),
    )


def _find_schema() -> tuple[Schema, ...]:
    def b1(rng, a): return [Action.make("file_search", root=a["path"], query=a["query"])]
    def b2(rng, a): return [Action.make("grep", root=a["path"], pattern=a["query"])]
    def b3(rng, a): return [Action.make("index_search", root=a["path"], query=a["query"])]
    return (
        Schema(("find {query} in {path}", "{path} altında {query} ara",
                "{path} içinde {query} bul", "search {query} under {path}"),
               obj="file", scope="path", placeholders=("path", "query"), build=b1),
        Schema(("grep {query} in {path}", "{path} grep {query}",
                "{query} içeren dosyaları bul {path}", "grep -r {query} {path}"),
               obj="file", scope="path", placeholders=("path", "query"), build=b2),
        Schema(("index search {query} in {path}", "indeks ara {query} {path}"),
               obj="file", scope="path", placeholders=("path", "query"), build=b3),
    )


def _count_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("sh", cmd=f"find {a['path']} -name '*.{a['ext']}' | wc -l", timeout=15)]
    return (
        Schema(("count {ext} files in {path}", "{path} altındaki {ext} sayısını bul",
                "{path} içinde kaç {ext} var", "how many {ext} files in {path}"),
               obj="file", scope="path", placeholders=("path", "ext"), build=b),
    )


def _disk_schema() -> tuple[Schema, ...]:
    def b1(rng, a): return [Action.make("disk_usage", path=a["path"])]
    def b2(rng, a): return [Action.make("sh", cmd=f"du -sh {a['path']}", timeout=15)]
    return (
        Schema(("report disk usage at {path}", "{path} disk raporu", "{path} disk kullanımı",
                "rapor disk {path}", "disk usage of {path}"),
               obj="disk", scope="path", placeholders=("path",), build=b1),
        Schema(("du {path}", "{path} ne kadar yer", "size of {path}"),
               obj="disk", scope="path", placeholders=("path",), build=b2),
    )


def _summarize_schema() -> tuple[Schema, ...]:
    def b(rng, a):
        return [
            Action.make("file_read", path=a["file"]),
            Action.make("summarize_text", text="<file>", max_lines=20),
        ]
    return (
        Schema(("summarize {file}", "{file} özetle", "özetle {file}",
                "summarise {file}", "{file} özet çıkar"),
               obj="file", scope="path", placeholders=("file",), build=b),
    )


def _git_status_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("git_status", repo=a["path"])]
    return (
        Schema(("git status in {path}", "{path} git durumu", "{path} repo status",
                "git status {path}", "{path} git status"),
               obj="repo", scope="path", placeholders=("path",), build=b),
    )


def _commit_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("sh", cmd="git -C {} commit -am 'auto commit'".format(a["path"]), timeout=20)]
    return (
        Schema(("commit changes in {path}", "{path} commit at",
                "git commit {path}", "kaydet repoya {path}"),
               obj="repo", scope="path", placeholders=("path",), build=b),
    )


def _organize_schema() -> tuple[Schema, ...]:
    def b(rng, a):
        return [
            Action.make("file_list", path=a["path"]),
            Action.make("mkdir", path=f"{a['path']}/sorted"),
        ]
    return (
        Schema(("organize folder {path}", "{path} düzenle", "tidy {path}",
                "organize {path}", "{path} sırala"),
               obj="folder", scope="path", placeholders=("path",), build=b),
    )


def _clean_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("file_list", path=a["path"], pattern="*.tmp")]
    return (
        Schema(("clean {path}", "{path} temizle", "purge {path}", "clear cache in {path}"),
               obj="folder", scope="path", placeholders=("path",), build=b),
    )


def _inspect_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("file_read", path=a["file"], max_bytes=4000)]
    return (
        Schema(("inspect {file}", "{file} incele", "incele {file}",
                "examine {file}", "{file} aç bak"),
               obj="file", scope="path", placeholders=("file",), build=b),
    )


def _download_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("http_download", url=a["url"], dst="/tmp/dl.bin")]
    return (
        Schema(("download {url}", "{url} indir", "fetch {url}", "{url} çek", "get {url}"),
               obj="url", scope="url", placeholders=("url",), build=b),
    )


def _research_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("http_get", url=a["url"])]
    return (
        Schema(("research {url}", "{url} araştır", "fetch info from {url}",
                "{url} hakkında ara", "search {url}"),
               obj="url", scope="url", placeholders=("url",), build=b),
    )


def _run_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("sh", cmd=f"python {a['file']}", timeout=30)]
    return (
        Schema(("run {file}", "{file} çalıştır", "execute {file}",
                "başlat {file}", "{file} koş"),
               obj="code", scope="path", placeholders=("file",), build=b),
    )


def _watch_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("process_list", pattern=a["process"])]
    return (
        Schema(("watch {process}", "{process} izle", "monitor {process}",
                "{process} takip et", "follow {process}"),
               obj="process", scope="path", placeholders=("process",), build=b),
    )


def _delete_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("file_delete", path=a["file"], confirmed=True)]
    return (
        Schema(("delete {file}", "{file} sil", "rm {file}", "{file} kaldır", "remove {file}"),
               obj="file", scope="path", placeholders=("file",), build=b),
    )


def _move_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("file_move", src=a["src"], dst=a["dst"])]
    return (
        Schema(("move {src} to {dst}", "{src} {dst} taşı", "mv {src} {dst}",
                "{src} dan {dst} taşı"),
               obj="file", scope="path", placeholders=("src", "dst"), build=b),
    )


def _copy_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("file_copy", src=a["src"], dst=a["dst"])]
    return (
        Schema(("copy {src} to {dst}", "{src} {dst} kopyala", "cp {src} {dst}"),
               obj="file", scope="path", placeholders=("src", "dst"), build=b),
    )


def _rename_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("file_move", src=a["src"], dst=a["dst"])]
    return (
        Schema(("rename {src} to {dst}", "{src} adını {dst} yap",
                "{src} yeniden adlandır {dst}"),
               obj="file", scope="path", placeholders=("src", "dst"), build=b),
    )


def _build_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("sh", cmd=f"make -C {a['path']}", timeout=60)]
    return (
        Schema(("build {path}", "{path} derle", "compile {path}", "{path} build"),
               obj="code", scope="path", placeholders=("path",), build=b),
    )


def _test_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("sh", cmd=f"pytest {a['path']}", timeout=120)]
    return (
        Schema(("test {path}", "{path} test et", "run tests in {path}",
                "pytest {path}", "{path} testleri çalıştır"),
               obj="test", scope="path", placeholders=("path",), build=b),
    )


def _lint_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("sh", cmd=f"ruff check {a['path']}", timeout=30)]
    return (
        Schema(("lint {path}", "{path} lint", "check {path}", "{path} kontrol et",
                "{path} ruff"),
               obj="code", scope="path", placeholders=("path",), build=b),
    )


def _index_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("index_search", root=a["path"], query=a["query"])]
    return (
        Schema(("index {path} for {query}", "{path} indeksle {query}",
                "{path} içinde indeks {query}"),
               obj="folder", scope="path", placeholders=("path", "query"), build=b),
    )


def _extract_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("archive_extract", src=a["file"], dst=a["path"])]
    return (
        Schema(("extract {file} to {path}", "{file} çıkar {path}",
                "unpack {file} into {path}"),
               obj="file", scope="path", placeholders=("file", "path"), build=b),
    )


def _convert_schema() -> tuple[Schema, ...]:
    def b(rng, a):
        return [Action.make("sh", cmd=f"pandoc {a['src']} -o {a['dst']}", timeout=30)]
    return (
        Schema(("convert {src} to {dst}", "{src} dönüştür {dst}",
                "convert {src} into {dst}"),
               obj="file", scope="path", placeholders=("src", "dst"), build=b),
    )


def _share_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("http_post", url=a["url"], body=f"file={a['file']}")]
    return (
        Schema(("share {file} via {url}", "{file} {url} paylaş",
                "post {file} to {url}", "{file} gönder {url}"),
               obj="file", scope="url", placeholders=("file", "url"), build=b),
    )


def _report_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("disk_usage", path=a["path"])]
    return (
        Schema(("report disk usage here", "rapor disk burada", "disk durumu rapor",
                "report disk {path}", "{path} disk raporu çıkar"),
               obj="disk", scope="path", placeholders=("path",), build=b),
    )


def _help_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("noop")]
    return (
        Schema(("help", "yardım", "?", "yardim"),
               obj="", scope="", placeholders=(), build=b),
    )


def _write_code_schema() -> tuple[Schema, ...]:
    def b(rng, a):
        return [Action.make("file_write", path=a["file"], content="# generated\n", overwrite=True)]
    return (
        Schema(("write code for {file}", "{file} kod yaz", "{file} implement et",
                "implement {file}", "{file} için kod yaz"),
               obj="code", scope="path", placeholders=("file",), build=b),
    )


def _make_image_schema() -> tuple[Schema, ...]:
    def b(rng, a): return [Action.make("noop")]
    return (
        Schema(("make image of {query}", "{query} görsel üret",
                "generate image of {query}", "{query} çiz"),
               obj="image", scope="path", placeholders=("query",), build=b),
    )


SCHEMA_BANK: dict[str, tuple[Schema, ...]] = {
    "list": _list_schema(),
    "find": _find_schema(),
    "count": _count_schema(),
    "report": _report_schema() + _disk_schema(),
    "summarize": _summarize_schema(),
    "git_status": _git_status_schema(),
    "commit": _commit_schema(),
    "organize": _organize_schema(),
    "clean": _clean_schema(),
    "inspect": _inspect_schema(),
    "download": _download_schema(),
    "research": _research_schema(),
    "run": _run_schema(),
    "watch": _watch_schema(),
    "delete": _delete_schema(),
    "move": _move_schema(),
    "copy": _copy_schema(),
    "rename": _rename_schema(),
    "build": _build_schema(),
    "test": _test_schema(),
    "lint": _lint_schema(),
    "index": _index_schema(),
    "extract": _extract_schema(),
    "convert": _convert_schema(),
    "share": _share_schema(),
    "help": _help_schema(),
    "write_code": _write_code_schema(),
    "make_image": _make_image_schema(),
}


# Negative pairs: same intent verb, but the action is wrong primitive. These
# get reward=0 so the reward-conditioned head learns to suppress them.
NEGATIVE_PRIMITIVES: tuple[str, ...] = ("noop", "now_iso", "uname", "cpu_info")


def _render_template(template: str, args: dict[str, str]) -> str:
    out = template
    for k, v in args.items():
        out = out.replace("{" + k + "}", v)
    return out


def synthesize_corpus(
    *,
    examples_per_verb: int = 100,
    seed: int = 0,
    negative_ratio: float = 0.1,
) -> list[SeedExample]:
    """Deterministically generate (intent, actions, reward) examples.

    `examples_per_verb` positives per verb (rounded across schemas) plus
    ~negative_ratio negatives anywhere.
    """
    rng = random.Random(seed)
    out: list[SeedExample] = []
    for verb, schemas in SCHEMA_BANK.items():
        per_schema = max(1, examples_per_verb // max(1, len(schemas)))
        for schema in schemas:
            for _ in range(per_schema):
                args: dict[str, str] = {ph: PLACEHOLDER_FNS[ph](rng) for ph in schema.placeholders}
                template = rng.choice(schema.raw_templates)
                raw = _render_template(template, args)
                intent_args = tuple((k, v) for k, v in args.items())
                intent = Intent(
                    verb=verb,
                    obj=schema.obj,
                    scope=schema.scope,
                    args=intent_args,
                    raw=raw,
                    confidence=1.0,
                )
                actions = tuple(schema.build(rng, args))
                out.append(SeedExample(intent=intent, actions=actions, reward=0.85, source="synth"))

    # negatives — copy random positive intents, replace actions with noop/wrong
    n_neg = int(len(out) * negative_ratio)
    for _ in range(n_neg):
        base = rng.choice(out)
        wrong = Action.make(rng.choice(NEGATIVE_PRIMITIVES))
        out.append(
            SeedExample(
                intent=base.intent,
                actions=(wrong,),
                reward=0.05,
                source="synth_neg",
            )
        )

    rng.shuffle(out)
    return out
