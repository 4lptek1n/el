from el.intent import Intent
from el.primitives import Action
from el.transformer.tokenizer import ActionTokenizer, reward_bucket, reward_token


def test_reward_buckets():
    assert reward_bucket(0.0) == 0
    assert reward_bucket(1.0) == 4
    assert reward_token(0.5) == "<reward:2>"


def test_tokenizer_encode_roundtrip():
    tok = ActionTokenizer.build(["list this folder", "fetch example"], word_cap=256)
    intent = Intent(verb="list", scope="this_folder", raw="list this folder")
    actions = [Action.make("file_list", path=".")]
    ids = tok.encode_row(intent.to_dict(), [a.to_dict() for a in actions], reward=0.8)
    assert ids[0] == tok.tid("<bos>")
    assert ids[-1] == tok.tid("<eos>")
    decoded = tok.decode(ids)
    assert "<verb:list>" in decoded
    assert "<prim:file_list>" in decoded


def test_tokenizer_vocab_has_verbs_and_primitives():
    tok = ActionTokenizer.build([], word_cap=32)
    assert tok.tid("<verb:list>") >= 0
    assert tok.tid("<prim:sh>") >= 0
    assert tok.tid("<pad>") == 0


def test_tokenizer_save_load(tmp_path):
    tok = ActionTokenizer.build(["alpha beta"], word_cap=16)
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = ActionTokenizer.load(path)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.tid("<verb:list>") == tok.tid("<verb:list>")
