from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
import yaml
from tqdm import tqdm

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

NUM_CLASSES = 2
SAFE, UNSAFE = 0, 1

ON_KAGGLE = Path("/kaggle").exists()
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"


@dataclass(frozen=True)
class Config:
    model_name: str
    max_seq_length: int
    fp16: bool
    batch_size: int

    hidden: int
    dropout: float
    lr: float
    weight_decay: float
    max_epochs: int
    checkpoint_every: int

    data_dir: Path
    mask_emails: bool
    email_placeholder: str
    seed: int
    expected_counts: dict | None

    variant: str
    instruct_prefix: str

    @classmethod
    def load(cls, path: Path) -> "Config":

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        m, h, d = raw["model"], raw["head"], raw["data"]
        counts = d.get("expected_counts")
        every = int(h["checkpoint_every"])
        if every < 1:
            raise ValueError(f"head.checkpoint_every must be >= 1, got {every}")
        return cls(
            model_name=str(m["name"]),
            max_seq_length=int(m["max_seq_length"]),
            fp16=bool(m["fp16"]),
            batch_size=int(m["batch_size"]),
            hidden=int(h["hidden"]),
            dropout=float(h["dropout"]),
            lr=float(h["lr"]),
            weight_decay=float(h["weight_decay"]),
            max_epochs=int(h["max_epochs"]),
            checkpoint_every=every,
            data_dir=(SCRIPT_DIR / str(d["dir"])).resolve(),
            mask_emails=bool(d["mask_emails"]),
            email_placeholder=str(d["email_placeholder"]),
            seed=int(d["seed"]),
            expected_counts=({k: tuple(v) for k, v in counts.items()} if counts else None),
            variant=str(raw["variant"]),
            instruct_prefix=str(raw["instruct_prefix"]),
        )

    def to_json(self) -> dict:
        return {
            "model_name": self.model_name, "max_seq_length": self.max_seq_length,
            "chunking": False, "fp16": self.fp16, "batch_size": self.batch_size,
            "hidden": self.hidden, "dropout": self.dropout, "lr": self.lr,
            "weight_decay": self.weight_decay, "max_epochs": self.max_epochs,
            "checkpoint_every": self.checkpoint_every, "num_classes": NUM_CLASSES,
            "mask_emails": self.mask_emails, "seed": self.seed, "variant": self.variant,
            "instruct_prefix": self.instruct_prefix if self.variant == "instruct" else None,
        }


def load_data(split: str, cfg: Config) -> tuple[list[str], np.ndarray, list[str] | None]:
    path = cfg.data_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run prepare_head_data.py first:\n"
            f"    python {SCRIPT_DIR.parent / 'dataset_construction' / 'scripts' / 'prepare_head_data.py'}"
        )
    df = pd.read_parquet(path)

    n_unsafe, n_safe = int(df["label"].sum()), int((1 - df["label"]).sum())
    print(f"  {split:5s}  {len(df):5d} rows  {n_unsafe:4d} unsafe / {n_safe:4d} safe   <- {path}")
    if cfg.expected_counts:
        assert (n_unsafe, n_safe) == cfg.expected_counts[split], \
            (f"{split}: label counts {(n_unsafe, n_safe)} != expected "
             f"{cfg.expected_counts[split]} -- set data.expected_counts to null in the config "
             f"if you are deliberately using different data")

    ids_path = cfg.data_dir / f"{split}_node_ids.txt"
    node_ids = ids_path.read_text(encoding="utf-8").splitlines() if ids_path.exists() else None
    if node_ids is not None and len(node_ids) != len(df):
        raise ValueError(f"{ids_path.name} has {len(node_ids)} ids for {len(df)} rows")

    return df["text"].astype(str).tolist(), df["label"].to_numpy(), node_ids


def prepare_texts(texts: list[str], cfg: Config) -> list[str]:
    if cfg.mask_emails:
        texts = [EMAIL_RE.sub(cfg.email_placeholder, t) for t in texts]
    if cfg.variant == "instruct":
        texts = [cfg.instruct_prefix + t for t in texts]
    return texts

class SafetyClassifier(nn.Module):
    def __init__(self, cfg: Config, freeze: bool = True):
        super().__init__()


        self.encoder = SentenceTransformer(cfg.model_name)
        self.encoder.max_seq_length = cfg.max_seq_length
        self.batch_size = cfg.batch_size

        dim = (self.encoder.get_embedding_dimension()
               if hasattr(self.encoder, "get_embedding_dimension")
               else self.encoder.get_sentence_embedding_dimension())
        self.dim = dim

        self.head = nn.Linear(dim, NUM_CLASSES) if cfg.hidden == 0 else nn.Sequential(
            nn.Linear(dim, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, NUM_CLASSES),
        )

        self.frozen = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        n_frozen = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        n_train = sum(p.numel() for p in self.head.parameters() if p.requires_grad) / 1e3
        print(f"  encoder {n_frozen:.0f}M params ({'frozen' if freeze else 'TRAINABLE'})  "
              f"head {n_train:.0f}K params (trainable)  dim={dim}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def tokenize(self, texts: list[str]) -> dict:
        fn = getattr(self.encoder, "preprocess", None) or self.encoder.tokenize
        return fn(texts)

    def embed(self, features: dict):
        return self.encoder(features)["sentence_embedding"].float()

    def forward(self, features: dict):
        return self.head(self.embed(features))


def to_device(features: dict, device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in features.items()}


@torch.no_grad()
def predict_proba(model: SafetyClassifier, texts: list[str], device, desc: str) -> np.ndarray:

    model.eval()
    out = []
    for i in tqdm(range(0, len(texts), model.batch_size), desc=desc, leave=True):
        feats = to_device(model.tokenize(texts[i: i + model.batch_size]), device)
        probs = model(feats).softmax(dim=1)[:, UNSAFE]
        out.append(probs.cpu().numpy())
    return np.concatenate(out)

def save_checkpoint(model: SafetyClassifier, cfg: Config, epoch: int, loss: float,
                    ckpt_dir: Path) -> Path:
    path = ckpt_dir / f"head_epoch{epoch:02d}.pt"
    torch.save({"state_dict": model.head.state_dict(), "hidden": cfg.hidden,
                "dim": model.dim, "num_classes": NUM_CLASSES, "variant": cfg.variant,
                "model_name": cfg.model_name, "epoch": epoch, "train_loss": float(loss)}, path)
    return path


def train_head(model: SafetyClassifier, device, train_texts: list[str], y_tr: np.ndarray,
               cfg: Config, ckpt_dir: Path) -> list[dict]:

    torch.manual_seed(cfg.seed)

    counts = np.bincount(y_tr, minlength=NUM_CLASSES)
    weights = torch.tensor(len(y_tr) / (NUM_CLASSES * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64)).to(device)

    print(f"  class weights: safe={weights[SAFE]:.2f} unsafe={weights[UNSAFE]:.2f}  device={device}")
    print(f"  checkpoint every {cfg.checkpoint_every} epoch(s) -> {ckpt_dir}")

    rng = np.random.default_rng(cfg.seed)
    history: list[dict] = []

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(train_texts))
        running = 0.0
        for i in tqdm(range(0, len(order), cfg.batch_size), desc=f"epoch {epoch}", leave=False):
            idx = order[i: i + cfg.batch_size]
            feats = to_device(model.tokenize([train_texts[j] for j in idx]), device)
            opt.zero_grad()
            loss = loss_fn(model(feats), y_tr_t[idx])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)

        epoch_loss = running / len(order)
        is_last = epoch == cfg.max_epochs
        ckpt = (save_checkpoint(model, cfg, epoch, epoch_loss, ckpt_dir)
                if (epoch % cfg.checkpoint_every == 0 or is_last) else None)

        secs = time.time() - t0
        print(f"  epoch {epoch:2d}  loss {epoch_loss:.4f}  ({secs:.0f}s)"
              f"{'  -> ' + ckpt.name if ckpt else ''}")
        history.append({"epoch": epoch, "loss": epoch_loss, "seconds": secs,
                        "checkpoint": ckpt.name if ckpt else None})

    n_ckpt = sum(1 for h in history if h["checkpoint"])
    print(f"  done: {cfg.max_epochs} epochs, {n_ckpt} checkpoints saved")
    return history
