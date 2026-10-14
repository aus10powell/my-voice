from pathlib import Path

from my_voice.eval.librispeech import Utterance, _index_subset


def test_index_subset_reads_trans_txt(tmp_path: Path):
    chapter = tmp_path / "84" / "121123"
    chapter.mkdir(parents=True)
    utt = "84-121123-0000"
    (chapter / f"{utt}.wav").write_bytes(b"RIFF")
    (chapter / "84-121123.trans.txt").write_text(
        f"{utt} HELLO WORLD\n84-121123-0001 MISSING AUDIO\n",
        encoding="utf-8",
    )
    found = _index_subset(tmp_path, max_utterances=10)
    assert len(found) == 1
    assert isinstance(found[0], Utterance)
    assert found[0].utt_id == utt
    assert found[0].reference == "HELLO WORLD"
