from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTokenizer:
    pad_token_id: int = 256
    bos_token_id: int = 257
    eos_token_id: int = 258
    action_token_id: int = 259
    tool_call_token_id: int = 260
    tool_result_token_id: int = 261
    tool_end_token_id: int = 262
    thought_token_id: int = 263

    _SPECIAL_MIN: int = 256
    _SPECIAL_MAX: int = 263

    def is_special(self, token_id: int) -> bool:
        return self._SPECIAL_MIN <= token_id <= self._SPECIAL_MAX

    def encode_text(self, text: str, max_len: int, add_bos_eos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8", errors="replace"))[: max_len - (2 if add_bos_eos else 0)]
        if add_bos_eos:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def encode_mixed(self, parts: list[str | int], max_len: int, add_bos_eos: bool = True) -> list[int]:
        """Interleave UTF-8 byte fragments (0-255) with special token IDs (256-263)."""
        ids: list[int] = []
        tail = 1 if add_bos_eos else 0
        limit = max_len - tail - (1 if add_bos_eos else 0)

        if add_bos_eos:
            ids.append(self.bos_token_id)

        for part in parts:
            if len(ids) >= max_len - tail:
                break
            if isinstance(part, int):
                if not self.is_special(part):
                    raise ValueError(f"not a special token id: {part}")
                ids.append(part)
                continue
            for byte in part.encode("utf-8", errors="replace"):
                if len(ids) >= max_len - tail:
                    break
                ids.append(byte)

        if add_bos_eos:
            ids.append(self.eos_token_id)
        return ids[:max_len]

    def pad_batch(self, rows: list[list[int]], pad_id: int | None = None) -> tuple[list[list[int]], int]:
        pad = self.pad_token_id if pad_id is None else pad_id
        max_len = max(len(r) for r in rows)
        padded = [r + [pad] * (max_len - len(r)) for r in rows]
        return padded, max_len
