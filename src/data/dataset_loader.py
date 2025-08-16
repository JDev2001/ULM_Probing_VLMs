from datasets import load_dataset
from functools import partial
import random
import re
import string
from typing import List, Tuple, Set

MSCocoCategories = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle"
]

_punct_re = re.compile(f"[{re.escape(string.punctuation)}]")

def _normalize(text: str) -> str:
    return _punct_re.sub(" ", text.lower())

def _tokens_and_ngrams(text: str, max_n: int = 3) -> Set[str]:
    words = [w for w in _normalize(text).split() if w]
    out = set(words)
    for n in range(2, max_n + 1):
        for i in range(len(words) - n + 1):
            out.add(" ".join(words[i:i + n]))
    return out

def _batch_neg_captions(examples, indices, neg_captions):
    pos = examples["caption"]
    idxs = indices
    return {
        "caption_pos": pos,
        "caption_neg": [neg_captions[i] for i in idxs],
        "url": examples["url"],
    }

def _batch_categories(examples, categories: List[str]):
    captions = examples["caption_pos"] if "caption_pos" in examples else examples["caption"]
    pos_list = []
    neg_list = []
    for t in captions:
        toks = _tokens_and_ngrams(t)
        pos = [c for c in categories if c in toks]
        neg = [c for c in categories if c not in toks]
        pos_list.append(pos)
        neg_list.append(neg)
    return {"pos_categories": pos_list, "neg_categories": neg_list}

class DSLoader:
    def __init__(self, split: str, num_proc: int = None, batch_size: int = 2000, seed: int = 42):
        self.split = split
        self.num_proc = num_proc
        self.batch_size = batch_size
        self.seed = seed
        self.ds = load_dataset("cat-state/mscoco-1st-caption", split=split)
        keep_cols = [c for c in ["url", "caption"] if c in self.ds.column_names]
        if keep_cols and set(keep_cols) != set(self.ds.column_names):
            self.ds = self.ds.remove_columns([c for c in self.ds.column_names if c not in keep_cols])

        self._neg_captions = None

    def _prepare_negative_captions(self):
        if self._neg_captions is not None:
            return
        captions = self.ds["caption"]
        rng = random.Random(self.seed)
        idxs = list(range(len(captions)))
        rng.shuffle(idxs)
        shifted = idxs[1:] + idxs[:1]
        neg = [captions[j] for j in shifted]
        for k in range(len(captions)):
            if neg[k] == captions[k]:
                j = (k + 2) % len(captions)
                neg[k] = captions[j]
        self._neg_captions = neg

    def get_caption_ds(self):
        self._prepare_negative_captions()
        fn = partial(_batch_neg_captions, neg_captions=self._neg_captions)
        ds2 = self.ds.map(
            fn,
            batched=True,
            with_indices=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            load_from_cache_file=True,
            keep_in_memory=False,
            desc="building captions",
        )
        cols = ["url", "caption_pos", "caption_neg"]
        keep = [c for c in cols if c in ds2.column_names]
        return ds2.select_columns(keep)

    def get_category_ds(self, categories: List[str] = None):
        if categories is None:
            categories = MSCocoCategories
        fn = partial(_batch_categories, categories=categories)
        ds3 = self.ds.map(
            fn,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            load_from_cache_file=True,
            keep_in_memory=False,
            desc="building categories",
        )
        cols = ["url", "caption_pos", "caption_neg", "pos_categories", "neg_categories"]
        keep = [c for c in cols if c in ds3.column_names]
        return ds3.select_columns(keep)
