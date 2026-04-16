"""Token scheme for action-grounded sequences.

Vocabulary is constructed deterministically from:

1. Special tokens (bos, eos, sep, pad, unk, reward buckets).
2. Verb tokens (one per canonical verb in parser.VERBS).
3. Object tokens (scope/obj vocabulary).
4. Primitive tokens (one per primitive name).
5. A byte-pair-style fallback for free-text arg values (kept small: words
   split by whitespace with a fixed-size vocabulary cap).

A training row is encoded as:

    <bos> <cmd> <verb:V> <obj:O> <scope:S> <arg_k> <arg_v> ...
          <sep> <act> <prim:P1> <arg_k1> <arg_v1> ...
          <sep> <act> <prim:P2> ...
          <sep> <reward:bucket> <eos>
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..parser import VERBS
from ..primitives import PRIMITIVES


SPECIAL_TOKENS: tuple[str, ...] = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<cmd>",
    "<act>",
    "<arg_k>",
    "<arg_v>",
    "<reward:0>",
    "<reward:1>",
    "<reward:2>",
    "<reward:3>",
    "<reward:4>",
)


REWARD_BUCKETS = 5


def reward_bucket(reward: float) -> int:
    if reward <= 0:
        return 0
    if reward >= 1:
        return REWARD_BUCKETS - 1
    return min(REWARD_BUCKETS - 1, int(reward * REWARD_BUCKETS))


def reward_token(reward: float) -> str:
    return f"<reward:{reward_bucket(reward)}>"


@dataclass
class ActionTokenizer:
    word_cap: int = 4096
    max_len: int = 512
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, texts: Iterable[str] = (), word_cap: int = 4096, max_len: int = 512) -> "ActionTokenizer":
        tok = cls(word_cap=word_cap, max_len=max_len)
        tok._install_specials()
        for spec in VERBS:
            tok._add(f"<verb:{spec.canonical}>")
        for obj in (
            "pdf", "image", "video", "folder", "file", "repo", "url",
            "code", "note", "report", "paper", "process", "disk", "test", "",
        ):
            tok._add(f"<obj:{obj or '_'}>")
        for scope in ("this_folder", "downloads", "home", "path", "url", ""):
            tok._add(f"<scope:{scope or '_'}>")
        for name in PRIMITIVES:
            tok._add(f"<prim:{name}>")
        word_counts: dict[str, int] = {}
        for text in texts:
            for word in _simple_words(text):
                word_counts[word] = word_counts.get(word, 0) + 1
        ordered = sorted(word_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for word, _ in ordered[:word_cap]:
            tok._add(f"w:{word}")
        return tok

    def _install_specials(self) -> None:
        for t in SPECIAL_TOKENS:
            self._add(t)

    def _add(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        new_id = len(self.id_to_token)
        self.token_to_id[token] = new_id
        self.id_to_token.append(token)
        return new_id

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def tid(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["<unk>"])

    def encode_command(self, intent_dict: dict) -> list[int]:
        out: list[int] = [self.tid("<bos>"), self.tid("<cmd>")]
        out.append(self.tid(f"<verb:{intent_dict.get('verb', '')}>"))
        obj = intent_dict.get("obj") or "_"
        out.append(self.tid(f"<obj:{obj}>"))
        scope = intent_dict.get("scope") or "_"
        out.append(self.tid(f"<scope:{scope}>"))
        for key, value in intent_dict.get("args") or []:
            out.append(self.tid("<arg_k>"))
            for w in _simple_words(str(key)):
                out.append(self.tid(f"w:{w}"))
            out.append(self.tid("<arg_v>"))
            for w in _simple_words(str(value)):
                out.append(self.tid(f"w:{w}"))
        return out

    def encode_actions(self, actions: list[dict]) -> list[int]:
        out: list[int] = []
        for action in actions:
            out.append(self.tid("<sep>"))
            out.append(self.tid("<act>"))
            out.append(self.tid(f"<prim:{action.get('name')}>"))
            for key, value in action.get("kwargs") or []:
                out.append(self.tid("<arg_k>"))
                for w in _simple_words(str(key)):
                    out.append(self.tid(f"w:{w}"))
                out.append(self.tid("<arg_v>"))
                for w in _simple_words(str(value))[:8]:
                    out.append(self.tid(f"w:{w}"))
        return out

    def encode_row(self, intent_dict: dict, actions: list[dict], reward: float) -> list[int]:
        ids = self.encode_command(intent_dict)
        ids += self.encode_actions(actions)
        ids.append(self.tid("<sep>"))
        ids.append(self.tid(reward_token(reward)))
        ids.append(self.tid("<eos>"))
        return ids[: self.max_len]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] if 0 <= i < len(self.id_to_token) else "<unk>" for i in ids]

    def save(self, path) -> None:
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {"word_cap": self.word_cap, "max_len": self.max_len, "id_to_token": self.id_to_token},
                fh,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path) -> "ActionTokenizer":
        import json

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        tok = cls(word_cap=data["word_cap"], max_len=data["max_len"])
        tok.id_to_token = list(data["id_to_token"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        return tok


def _simple_words(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in ".-_":
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out[:32]
