from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator

import torch

from .byte_tokenizer import ByteTokenizer

NUM_TOOLS = 64
_FUNC_MARKER = "<functioncall>"
_ASSISTANT_MARKERS = ("\nASSISTANT: ", "\nASSISTANT:", "ASSISTANT: ")


def fn_to_tool_id(name: str, num_tools: int = NUM_TOOLS) -> int:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_tools


def _extract_fn_name(func_json: str) -> str | None:
    try:
        obj = json.loads(func_json)
        name = obj.get("name")
        if isinstance(name, str) and name:
            return name
    except json.JSONDecodeError:
        return None


def _extract_functioncall_span(chat: str) -> tuple[int, int, str] | None:
    start = chat.find(_FUNC_MARKER)
    if start < 0:
        return None
    json_start = start + len(_FUNC_MARKER)
    while json_start < len(chat) and chat[json_start].isspace():
        json_start += 1
    if json_start >= len(chat) or chat[json_start] != "{":
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(json_start, len(chat)):
        ch = chat[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                payload = chat[json_start : i + 1]
                return start, i + 1, payload
    return None


def _assistant_action_position(chat: str, tok: ByteTokenizer, max_len: int) -> int:
    for marker in _ASSISTANT_MARKERS:
        idx = chat.find(marker)
        if idx < 0:
            continue
        prefix = chat[: idx + len(marker)]
        prefix_ids = tok.encode_mixed([prefix], max_len=max_len, add_bos_eos=True)
        pos = min(len(prefix_ids) - 1, max_len - 2)
        return max(pos, 1)
    return 1


def prepare_glaive_parts(chat: str, tok: ByteTokenizer) -> tuple[list[str | int], str | None]:
    span = _extract_functioncall_span(chat)
    if span is None:
        return [chat], None
    start, end, payload = span
    fn_name = _extract_fn_name(payload)
    parts: list[str | int] = [
        chat[:start],
        tok.tool_call_token_id,
        payload,
        tok.tool_end_token_id,
        chat[end:],
    ]
    return parts, fn_name


def attach_tool_labels(
    batch: dict[str, torch.Tensor],
    tok: ByteTokenizer,
    fn_name: str | None,
    chat: str,
) -> dict[str, torch.Tensor]:
    seq = batch["input_ids"]
    action_mask = torch.zeros_like(seq)
    tool_decision = torch.full_like(seq, -100)
    tool_id = torch.full_like(seq, -100)

    if fn_name is not None:
        tid = fn_to_tool_id(fn_name)
        positions = (seq[0] == tok.tool_call_token_id).nonzero(as_tuple=False).flatten()
        for pos in positions.tolist():
            action_mask[0, pos] = 1
            tool_decision[0, pos] = 1
            tool_id[0, pos] = tid
    else:
        pos = _assistant_action_position(chat, tok, seq.size(1))
        action_mask[0, pos] = 1
        tool_decision[0, pos] = 0

    batch["action_mask"] = action_mask
    batch["tool_decision_labels"] = tool_decision
    batch["tool_id_labels"] = tool_id
    return batch


def row_to_batch(row: dict[str, Any], tok: ByteTokenizer, seq_len: int) -> dict[str, torch.Tensor]:
    chat = row.get("chat") or row.get("text") or ""
    if not isinstance(chat, str) or not chat.strip():
        chat = "USER: ping\nASSISTANT: ok"
    parts, fn_name = prepare_glaive_parts(chat, tok)
    ids = tok.encode_mixed(parts, seq_len)
    t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    batch = {"input_ids": t, "labels": t.clone()}
    return attach_tool_labels(batch, tok, fn_name, chat)


def stream_glaive(
    tok: ByteTokenizer,
    seq_len: int,
    spec: Any | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    from datasets import load_dataset

    dataset_id = "glaiveai/glaive-function-calling-v2"
    split = "train"
    chat_key = "chat"
    if spec is not None:
        dataset_id = spec.dataset_id
        split = spec.split
        chat_key = getattr(spec, "chat_key", "chat")

    ds = load_dataset(dataset_id, split=split, streaming=True)
    for row in ds:
        if chat_key != "chat" and chat_key in row:
            row = {"chat": row[chat_key], **row}
        yield row_to_batch(row, tok, seq_len)
