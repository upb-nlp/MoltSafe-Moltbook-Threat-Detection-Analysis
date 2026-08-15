from __future__ import annotations

import logging
from typing import Dict
import torch
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


def _fix_uninitialized_position_ids(model) -> None:
    for module in model.modules():
        pos = getattr(module, "position_ids", None)
        if pos is None or pos.dim() != 1:
            continue
        expected = torch.arange(pos.size(0), device=pos.device, dtype=pos.dtype)
        if torch.equal(pos, expected):
            continue
        module.register_buffer("position_ids", expected, persistent=False)
        logger.warning(
            "Repaired uninitialised position_ids buffer (len=%d) on %s; "
            "transformers>=5 does not initialise non-persistent buffers.",
            pos.size(0), type(module).__name__,
        )


def _load_with_attn_fallback(model_name: str, device, trust_remote_code: bool, requested):

    ladder: list = []
    if requested:
        ladder.append(str(requested))
    ladder.append("sdpa")
    ladder.append(None)
    seen = set()
    ladder = [x for x in ladder if not (x in seen or seen.add(x))]

    last_err = None
    for impl in ladder:
        kwargs = {"device": device, "trust_remote_code": trust_remote_code}
        if impl is not None:
            kwargs["model_kwargs"] = {"attn_implementation": impl}
        try:
            model = SentenceTransformer(model_name, **kwargs)
            logger.info("attention backend: %s", impl or "<transformers default>")
            return model
        except (ImportError, ValueError, RuntimeError, TypeError, OSError) as exc:
            last_err = exc
            if impl is None:
                break
            logger.info("attn_implementation=%s unavailable (%s: %s); falling back",
                        impl, type(exc).__name__, str(exc).splitlines()[0][:120])
    raise RuntimeError(f"Could not load {model_name} with any attention backend") from last_err


def load_embedding_model(embed_cfg: Dict):
    model_name = embed_cfg["model_name"]
    logger.info("Loading model %s ...", model_name)
    model = _load_with_attn_fallback(
        model_name,
        device=embed_cfg.get("device"),
        trust_remote_code=bool(embed_cfg.get("trust_remote_code", False)),
        requested=embed_cfg.get("attn_implementation"),
    )
    _fix_uninitialized_position_ids(model)

    max_seq = embed_cfg.get("max_seq_length")
    if max_seq:
        model.max_seq_length = int(max_seq)
        logger.info("max_seq_length=%d", model.max_seq_length)

    if embed_cfg.get("fp16", False) and str(model.device).startswith("cuda"):
        model.half()
        logger.info("fp16 encoding ON (model.half() on %s)", model.device)
    return model
