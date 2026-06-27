from __future__ import annotations

import torch

from .byte_tokenizer import ByteTokenizer

HR_TEXT = "Dobar dan! Ovo je hrvatski primjer za miješani curriculum nakon warmup faze."
CODE_SNIPPET = "def hello(name: str) -> str:\n    return f'Hello, {name}!'\n"
STABLE_TEXT = (
    "HollowCore uči razmišljati u weights, a znanje dohvaća tool-call protokolom. "
    "Ovo je stabilan warmup tekst za prve korake treninga."
)


def _rows(tok: ByteTokenizer, text: str, seq_len: int) -> dict[str, torch.Tensor]:
    ids = tok.encode_text(text, seq_len)
    t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    return {"input_ids": t, "labels": t.clone()}


def view_b_for(category: str, view: str, tok: ByteTokenizer, seq_len: int) -> dict[str, torch.Tensor]:
    if view == "text_code":
        if category == "code":
            return _rows(tok, STABLE_TEXT, seq_len)
        return _rows(tok, CODE_SNIPPET, seq_len)
    if view == "paraphrase":
        return _rows(tok, HR_TEXT, seq_len)
    if view == "context_tool":
        parts: list[str | int] = [
            "USER: find docs\nASSISTANT: ",
            tok.tool_call_token_id,
            '{"name":"search","arguments":"{\\"q\\":\\"modal gpu\\"}"}',
            tok.tool_end_token_id,
            tok.tool_result_token_id,
            '{"hits":3}',
            tok.tool_end_token_id,
        ]
        ids = tok.encode_mixed(parts, seq_len)
        t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        return {"input_ids": t, "labels": t.clone()}
    return _rows(tok, STABLE_TEXT, seq_len)


def attach_view_b(base: dict[str, torch.Tensor], view_b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(base)
    out["input_ids_b"] = view_b["input_ids"]
    return out


def stream_multiview_pair(
    base: dict[str, torch.Tensor],
    category: str,
    view: str,
    tok: ByteTokenizer,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    vb = view_b_for(category, view, tok, seq_len)
    return attach_view_b(base, vb)
