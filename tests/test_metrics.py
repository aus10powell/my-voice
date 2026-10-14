from my_voice.eval.metrics import edit_distance, normalize_transcript, wer


def test_normalize_strips_punct():
    assert normalize_transcript("Hello, WORLD!") == ["hello", "world"]


def test_wer_perfect():
    assert wer("the cat sat", "the cat sat") == 0.0


def test_wer_substitution():
    assert wer("the cat sat", "the dog sat") == 1 / 3


def test_edit_distance_insert_delete():
    assert edit_distance(["a", "b"], ["a", "b", "c"]) == 1
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1
