# backend_core.py
import os
import re
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

pd.set_option("display.max_colwidth", 200)

# ===================== Config (paths) =====================
QURAN_PATH = os.getenv("QURAN_PATH", "Quran_data.xlsx")
MAQAS_PATH = os.getenv("MAQAS_PATH", "MAQAS.xlsx")
EMB_PATH   = os.getenv("EMB_PATH", "embeddings_bge_m3.npy")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index_bge_m3.idx")

# ===================== Columns =====================
AR_DIAC  = "Quran with diacritic"
AR_NOD   = "Quran without diacritic"
EN_COL   = "Translation in English"
SURA_COL = "Chapter"
AYAH_COL = "No verse in Chapter"
AR_EXACT = "fixed_words"

M_SURA_COL   = "Sura_No"
M_AYAH_COL   = "Verse_No"
M_NODIAC_COL = "Without_Diacritics"
M_TYPE_COL   = "Morph_Type"
M_GLOSS_COL  = "Gloss"

# ===================== Normalization =====================
AR_DIAC_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
TATWEEL_RE = re.compile(r"\u0640")

def normalize_ar(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = AR_DIAC_RE.sub("", s)
    s = TATWEEL_RE.sub("", s)
    s = s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    s = s.replace("ى","ي").replace("ة","ه")
    s = re.sub(r"[^\u0600-\u06FF\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_en(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_arabic(text: str) -> bool:
    return re.search(r"[\u0600-\u06FF]", str(text)) is not None

def vkey(sura: int, ayah: int) -> str:
    return f"{int(sura)}:{int(ayah)}"

try:
    print("BACKEND_CORE LOADED FROM:", __file__)
except Exception:
    print("BACKEND_CORE LOADED ✅")

# ===================== Debug logging =====================
DEBUG_SEARCH = os.getenv("DEBUG_SEARCH", "1") == "1"
def _log(msg: str):
    if DEBUG_SEARCH:
        print(msg, flush=True)

# ===================== Device =====================
_device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== Load data/index/models =====================
print("Loading data...")
df_verses = pd.read_excel(QURAN_PATH)
df_maqas  = pd.read_excel(MAQAS_PATH)

print("Loading embeddings/index...")
index = faiss.read_index(INDEX_PATH)

assert len(df_verses) == index.ntotal, "Mismatch df_verses vs faiss index!"
print("Verses:", len(df_verses), "| MAQAS rows:", len(df_maqas), "| Index:", index.ntotal)

# Pre-normalize verse text
verse_ar_norm       = df_verses[AR_NOD].astype(str).map(normalize_ar).tolist()          # full verse (for semantic / general)
verse_ar_exact_norm = df_verses[AR_EXACT].astype(str).map(normalize_ar).tolist()        # fixed_words (for exact match only)
verse_en_norm       = df_verses[EN_COL].astype(str).map(normalize_en).tolist()

# vkey maps
vkey_to_row: Dict[str, int] = {}
row_to_vkey: Dict[int, str] = {}
for i, r in df_verses.iterrows():
    vk = vkey(r[SURA_COL], r[AYAH_COL])
    vkey_to_row[vk] = int(i)
    row_to_vkey[int(i)] = vk

# ===================== Build MAQAS inverted index =====================
print("Building MAQAS indices...")
df_m = df_maqas.copy()
df_m["__morph"] = df_m[M_TYPE_COL].astype(str).str.lower()

# stem الحقيقي من Segmented_Word في صفوف Stem
df_m["__stem"] = ""
mask_stem = df_m["__morph"].str.fullmatch("stem", na=False)
df_m.loc[mask_stem, "__stem"] = df_m.loc[mask_stem, "Segmented_Word"].astype(str).map(normalize_ar)

# gloss
df_m["__gloss"] = df_m[M_GLOSS_COL].astype(str).map(normalize_en)
df_stem = df_m[mask_stem].copy()

verse_ar_tokens: Dict[str, set] = {}
verse_en_tokens: Dict[str, set] = {}

for _, r in df_stem.iterrows():
    vk = vkey(r[M_SURA_COL], r[M_AYAH_COL])
    tok_ar = r["__stem"]
    tok_en = r["__gloss"]

    if tok_ar:
        verse_ar_tokens.setdefault(vk, set()).add(tok_ar)

    if tok_en:
        s = verse_en_tokens.setdefault(vk, set())
        for w in tok_en.split():
            if len(w) >= 4:
                s.add(w)

inv_ar: Dict[str, set] = {}
inv_en: Dict[str, set] = {}

for vk, toks in verse_ar_tokens.items():
    for t in toks:
        inv_ar.setdefault(t, set()).add(vk)

for vk, toks in verse_en_tokens.items():
    for t in toks:
        inv_en.setdefault(t, set()).add(vk)

print("Arabic stem vocab:", len(inv_ar), "| English gloss vocab:", len(inv_en))

# ===================== Embedder + Reranker =====================
print("Loading embedder + reranker...")
embedder = SentenceTransformer("BAAI/bge-m3", device=_device)

reranker_name = "BAAI/bge-reranker-v2-m3"
tok_rr = AutoTokenizer.from_pretrained(reranker_name)
mdl_rr = AutoModelForSequenceClassification.from_pretrained(reranker_name).to(_device)
mdl_rr.eval()
print("Ready ✅ on device:", _device)










# ===================== Core helpers =====================
def faiss_candidate_ids(query_text: str, k_retrieve: int = 1800):
    q_emb = embedder.encode(query_text, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    if q_emb.ndim == 1:
        q_emb = q_emb[None, :]
    scores, ids = index.search(q_emb, min(k_retrieve, index.ntotal))
    ids = ids[0]
    scores = scores[0]
    mask = ids >= 0
    return ids[mask].astype(np.int64), scores[mask].astype(np.float32)

@torch.no_grad()
def rerank_bge(query: str, passages: List[str], batch_size=16, max_length=384) -> np.ndarray:
    scores = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i+batch_size]
        pairs = [[query, p] for p in batch]
        enc = tok_rr(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(_device) for k, v in enc.items()}
        out = mdl_rr(**enc)
        sc = out.logits.squeeze(-1).detach().float().cpu().numpy()
        scores.extend(sc.tolist())
    return np.array(scores, dtype=np.float32)

def build_passage(ix: int) -> str:
    row = df_verses.iloc[int(ix)]
    return f"{row[AR_DIAC]} [SEP] {row[EN_COL]}"

# ===================== MAQAS candidates + phrase hits =====================
def maqas_candidates_ar(query_ar: str) -> Tuple[set, List[str]]:
    q = normalize_ar(query_ar)
    if not q:
        return set(), []
    raw = [normalize_ar(x) for x in str(query_ar).split() if normalize_ar(x)]
    if not raw:
        raw = [q]
    toks = []
    for t in raw:
        toks.append(t)
        if t.startswith("ال") and len(t) > 2:
            toks.append(t[2:])
    seen = set()
    toks = [t for t in toks if not (t in seen or seen.add(t))]
    out = set()
    for t in toks[:6]:
        out |= inv_ar.get(t, set())
    return out, toks

def maqas_candidates_en(query_en: str) -> Tuple[set, List[str]]:
    toks = [t for t in normalize_en(query_en).split() if len(t) >= 4]
    if not toks:
        return set(), []
    out = set()
    for t in toks[:5]:
        out |= inv_en.get(t, set())
    return out, toks

def exact_phrase_hits_ar(phrase_ar: str) -> List[int]:
    ph = normalize_ar(phrase_ar)
    if not ph:
        return []
    return [i for i, txt in enumerate(verse_ar_exact_norm) if ph in txt]

def exact_phrase_hits_en(phrase_en: str) -> List[int]:
    ph = normalize_en(phrase_en)
    if not ph:
        return []
    return [i for i, txt in enumerate(verse_en_norm) if ph in txt]

def exact_word_hits_ar(word_ar: str) -> List[int]:
    w = normalize_ar(word_ar)
    if not w:
        return []
    needle = f" {w} "
    return [i for i, txt in enumerate(verse_ar_exact_norm) if needle in f" {txt} "]

def exact_word_hits_en(word_en: str) -> List[int]:
    w = normalize_en(word_en)
    if not w:
        return []
    needle = f" {w} "
    return [i for i, txt in enumerate(verse_en_norm) if needle in f" {txt} "]

# ===================== Main Search (NO expansions) =====================
def search_api(
    query: str,
    k_faiss: int = 1200,
    rerank_limit_non_guaranteed: int = 250,
    rerank_batch: int = 32
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    q = (query or "").strip()
    if not q:
        return [], {"error": "empty", "total": 0}

    ar_query = is_arabic(q)
    _log(f"[SEARCH] q='{q}' ar={ar_query}")

    # (1) guaranteed set (exact word/phrase + MAQAS hits) => lexical only
    if ar_query:
        maqas_vkeys, toks = maqas_candidates_ar(q)
        qn = normalize_ar(q)
        phrase_ids = exact_phrase_hits_ar(q) if " " in qn else exact_word_hits_ar(q)
    else:
        maqas_vkeys, toks = maqas_candidates_en(q)
        qn = normalize_en(q)
        phrase_ids = exact_phrase_hits_en(q) if " " in qn else exact_word_hits_en(q)

    maqas_ids = [vkey_to_row[vk] for vk in maqas_vkeys if vk in vkey_to_row]
    guaranteed_ids = sorted(set(maqas_ids) | set(phrase_ids))
    guaranteed_set = set(guaranteed_ids)

    _log(f"[LEX] maqas_vkeys={len(maqas_vkeys)} | maqas_ids={len(maqas_ids)} | "
         f"exact_ids={len(phrase_ids)} | guaranteed_ids={len(guaranteed_ids)}")

    SEM_USE_MAQAS_EXPANSION = os.getenv("SEM_USE_MAQAS_EXPANSION", "1") == "1"
    SEM_EXPAND_TOPN = int(os.getenv("SEM_EXPAND_TOPN", "8"))  # خله 6-12 ممتاز

    # (2) FAISS candidates pool (semantic candidates)
    embed_q = q
    if SEM_USE_MAQAS_EXPANSION and toks:
        toks_use = toks[:SEM_EXPAND_TOPN]
        embed_q = " | ".join([q] + toks_use)

    _log(f"[SEM-Q] embed_q='{embed_q}'")



    faiss_ids, faiss_scores = faiss_candidate_ids(embed_q, k_retrieve=k_faiss)
    id2fs = {int(i): float(s) for i, s in zip(faiss_ids.tolist(), faiss_scores.tolist())}

    other_part = [int(ix) for ix in faiss_ids.tolist() if int(ix) not in guaranteed_set]
    other_part.sort(key=lambda x: id2fs.get(int(x), -1e9), reverse=True)
    other_part = other_part[:rerank_limit_non_guaranteed]

    _log(f"[FAISS] k_faiss={k_faiss} | retrieved={len(faiss_ids)} | other_part={len(other_part)}")

    union_ids = sorted(set(guaranteed_ids) | set(other_part))
    _log(f"[UNION] union_ids={len(union_ids)}")

    if not union_ids:
        return [], {"error": "no candidates", "total": 0}

    # (3) reranker query
    if ar_query:
        rr_query = f"أوجد آيات في القرآن تتعلق بمفهوم: {q}. أعد الآيات المرتبطة معنى وسياقًا."
    else:
        rr_query = f"Find Quran verses that discuss the concept of: {q}. Return verses related by meaning and context."

    passages = [build_passage(ix) for ix in union_ids]
    rr_scores = rerank_bge(rr_query, passages, batch_size=rerank_batch, max_length=384)
    rr_map = {int(ix): float(sc) for ix, sc in zip(union_ids, rr_scores)}

    # (4) build rows + priority layers
    q_phrase = normalize_ar(q) if ar_query else normalize_en(q)
    rows = []

    for ix in union_ids:
        row = df_verses.iloc[int(ix)]
        vk = row_to_vkey[int(ix)]
        txt_norm = verse_ar_exact_norm[int(ix)] if ar_query else verse_en_norm[int(ix)]

        # exact phrase/word
        if " " in q_phrase:
            is_exact = int(q_phrase in txt_norm)
        else:
            is_exact = int(f" {q_phrase} " in f" {txt_norm} ")

        guaranteed = 1 if int(ix) in guaranteed_set else 0

        # priority: 3 exact, 1 guaranteed, 0 semantic
        if is_exact:
            priority = 3
        elif guaranteed:
            priority = 1
        else:
            priority = 0

        bucket = "lexical" if priority > 0 else "semantic"

        rows.append({
            "ix": int(ix),
            "ref": vk,
            "score_rr": float(rr_map.get(int(ix), -999.0)),
            "priority": int(priority),
            "bucket": bucket,
            "arabic": str(row[AR_DIAC]),
            "english": str(row[EN_COL]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["priority", "score_rr"], ascending=[False, False]).reset_index(drop=True)

    df_keep = df[df["priority"] > 0].copy()
    df_sem  = df[df["priority"] == 0].copy().sort_values("score_rr", ascending=False)

    _log(f"[BUCKETS] lexical(df_keep)={len(df_keep)} | semantic(df_sem_raw)={len(df_sem)}")

    # (5) semantic pool before LLM
    RR_MIN = -5.0
    TOP_SEM_TOTAL = int(os.getenv("TOP_SEM_TOTAL", "150"))  # 150 pool قبل LLM
    df_sem = df_sem[df_sem["score_rr"] >= RR_MIN].head(TOP_SEM_TOTAL)




    SEM_RETURN_TOPN = int(os.getenv("SEM_RETURN_TOPN", "10"))
    df_sem = df_sem.sort_values("score_rr", ascending=False)

    # (6) build df_sem_keep (NO LLM)
    df_sem_keep = df_sem.head(SEM_RETURN_TOPN).copy()

    df_final = pd.concat([df_keep, df_sem_keep], ignore_index=True)
    df_final = df_final.sort_values(
        ["priority", "score_rr"], ascending=[False, False]
    ).reset_index(drop=True)

    df_final.insert(0, "rank", np.arange(1, len(df_final) + 1))



    keep_cols = ["rank","ref","bucket","priority","score_rr","arabic","english"]

    results = df_final.reindex(columns=keep_cols, fill_value="").to_dict(orient="records")

    _log(f"[FINAL] total={len(df_final)} | lexical_total={len(df_keep)} | semantic_total={len(df_sem_keep)}")

    info = {
        "query": q,
        "ar_query": bool(ar_query),
        "total": int(len(df_final)),
        "lexical_total": int(len(df_keep)),
        "semantic_total": int(len(df_sem_keep)),
        "semantic_pool": int(len(df_sem)),
        "top_sem_total": int(TOP_SEM_TOTAL),
    }
    return results, info
