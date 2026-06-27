from __future__ import annotations

from typing import Any, Iterator

import torch

from .byte_tokenizer import ByteTokenizer
from .registry import DataSourceSpec, DatasetRegistry
from .stream_glaive import stream_glaive
from .stream_multiview import stream_multiview_pair


def _hf_text_stream(spec: DataSourceSpec) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(spec.dataset_id, spec.config, split=spec.split, streaming=True)
    for row in ds:
        val = row.get(spec.text_key) or ""
        if isinstance(val, str) and val.strip():
            yield val


def stream_packed_text(
    spec: DataSourceSpec,
    tok: ByteTokenizer,
    seq_len: int,
) -> Iterator[dict[str, torch.Tensor]]:
    """Token-packing: spaja vise dokumenata u punu seq_len sekvencu (bez praznog).

    Bez ovoga hf_text vraca jednu kratku sekvencu po dokumentu -> na 1M je
    sekvenca kratka, pa chunk-JEPA (treba >=2 chunka) cesto vrati nulu i ne
    trenira se nista na dugom kontekstu. Packing puni svaku sekvencu pravim
    tokenima preko vise dokumenata, odvojenih EOS-om.
    """
    buf: list[int] = []
    for text in _hf_text_stream(spec):
        buf.extend(text.encode("utf-8", errors="replace"))  # bajtovi 0-255 = tokeni
        buf.append(tok.eos_token_id)                         # razdjelnik izmedu dokumenata
        while len(buf) >= seq_len:
            sample = buf[:seq_len]
            buf = buf[seq_len:]
            t = torch.tensor(sample, dtype=torch.long).unsqueeze(0)
            yield {"input_ids": t, "labels": t.clone()}


def stream_category(
    category: str,
    spec: DataSourceSpec,
    tok: ByteTokenizer,
    seq_len: int,
) -> Iterator[dict[str, torch.Tensor]]:
    if spec.kind == "glaive":
        yield from stream_glaive(tok, seq_len, spec=spec)
        return

    yield from stream_packed_text(spec, tok, seq_len)


class MixedBatchIterator:
    def __init__(
        self,
        tok: ByteTokenizer,
        seq_len: int,
        sampler: Any,
        registry: DatasetRegistry | None = None,
        seed: int = 0,
    ) -> None:
        self.tok = tok
        self.seq_len = seq_len
        self.sampler = sampler
        self.registry = registry or DatasetRegistry.load()
        self.step = 0
        self._specs = dict(self.registry.sources)
        self._iters: dict[str, Iterator[dict[str, torch.Tensor]]] = {
            cat: iter(stream_category(cat, spec, tok, seq_len))
            for cat, spec in self._specs.items()
        }
        self.views = ("text_code", "paraphrase", "context_tool")
        self.seed = seed

    def _next_from(self, category: str) -> dict[str, torch.Tensor]:
        try:
            return next(self._iters[category])
        except StopIteration:
            spec = self._specs[category]
            self._iters[category] = iter(stream_category(category, spec, self.tok, self.seq_len))
            return next(self._iters[category])

    def next_batch(self) -> tuple[dict[str, torch.Tensor], str, dict[str, float]]:
        cat = self.sampler.pick_category(self.step)
        mix = self.sampler.mix_at_step(self.step)
        view = self.views[self.step % len(self.views)]
        base = self._next_from(cat)
        batch = stream_multiview_pair(base, cat, view, self.tok, self.seq_len)
        self.step += 1
        return batch, view, mix
