def load_flash_attn_func():
    try:
        from flash_attn import flash_attn_func
    except Exception as exc:
        raise RuntimeError(
            "FlashAttention-2 is required: install flash-attn on the Modal GPU image."
        ) from exc
    return flash_attn_func


def load_chunk_gdn2():
    try:
        from fla.ops.gdn2 import chunk_gdn2
    except Exception as exc:
        raise RuntimeError(
            "GDN-2 kernel is required: install flash-linear-attention[cuda] on the Modal image."
        ) from exc
    return chunk_gdn2


def assert_runtime_deps() -> None:
    load_flash_attn_func()
    load_chunk_gdn2()
