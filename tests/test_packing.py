"""Self-check za token-packing: puna seq_len sekvenca preko vise dokumenata."""

from __future__ import annotations

from data import streams
from data.byte_tokenizer import ByteTokenizer


def test_packing_fills_seq_len(monkeypatch) -> None:
    # lazni stream kratkih dokumenata (bez HF/mreze)
    monkeypatch.setattr(streams, "_hf_text_stream", lambda spec: iter(["abc def ghi"] * 500))
    tok = ByteTokenizer()
    seq_len = 128

    gen = streams.stream_packed_text(spec=None, tok=tok, seq_len=seq_len)
    sample = next(gen)

    # puna duljina (NE kratka jedan-dokument), svi tokeni u vokabularu, next-token LM
    assert tuple(sample["input_ids"].shape) == (1, seq_len)
    assert int(sample["input_ids"].max()) <= tok.eos_token_id
    assert int(sample["input_ids"].min()) >= 0
    assert sample["labels"].equal(sample["input_ids"])

    # drugi sample isto pun (leftover buffer nastavlja, nista se ne gubi)
    sample2 = next(gen)
    assert tuple(sample2["input_ids"].shape) == (1, seq_len)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
