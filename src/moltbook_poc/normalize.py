from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

EMAIL_PATTERN = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
EMAIL_RE = re.compile(EMAIL_PATTERN)

def mask_emails(text: str, placeholder: str = "@") -> str:
    return EMAIL_RE.sub(placeholder, text or "")

def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'[​-‏‪-‮⁠﻿]', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def dedup_by_hash(
    records: List[Dict],
    hash_key: str,
    score_key: Optional[str] = None,
) -> List[Dict]:
    seen: Dict[str, Dict] = {}
    for rec in records:
        h = rec[hash_key]
        if h not in seen:
            seen[h] = rec
        elif score_key is not None:
            if (rec.get(score_key) or 0) > (seen[h].get(score_key) or 0):
                seen[h] = rec
    return list(seen.values())
