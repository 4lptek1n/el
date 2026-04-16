from el.parser import Parser


def test_parses_list_english():
    intents = Parser().parse("list this folder")
    assert intents[0].verb == "list"
    assert intents[0].scope == "this_folder"


def test_parses_list_turkish():
    intents = Parser().parse("bu klasörü listele")
    assert intents[0].verb == "list"
    assert intents[0].scope == "this_folder"


def test_parses_summarize_pdf():
    intents = Parser().parse("summarize the pdfs in this folder")
    assert intents[0].verb == "summarize"
    assert intents[0].obj == "pdf"


def test_parses_summarize_turkish():
    intents = Parser().parse("bu klasördeki pdf'leri özetle")
    assert intents[0].verb == "summarize"
    assert intents[0].obj == "pdf"
    assert intents[0].scope == "this_folder"


def test_parses_url_arg():
    intents = Parser().parse("research https://example.com for recent papers")
    assert intents[0].verb == "research"
    assert intents[0].arg("url") == "https://example.com"


def test_unknown_command():
    intents = Parser().parse("zxqzxq blargh")
    assert intents[0].verb == "unknown"
    assert intents[0].confidence == 0.0


def test_git_status_multiword():
    intents = Parser().parse("show git status of this repo")
    verbs = {i.verb for i in intents}
    assert "git_status" in verbs


def test_canonical_key_stable():
    i1 = Parser().parse("list this folder")[0]
    i2 = Parser().parse("list this folder")[0]
    assert i1.canonical_key() == i2.canonical_key()
