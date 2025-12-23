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
    return re.search(r'[\u0600-\u06FF]', str(text)) is not None

def vkey(sura:int, ayah:int) -> str:
    return f"{int(sura)}:{int(ayah)}"

# ===================== Stopwords for expansions =====================
AR_STOP = set([
    "من","في","على","الى","إلى","عن","ما","لا","لم","لن","قد","ان","إن","أن","او","أو","ثم",
    "و","ف","ب","ك","ل","ال","هذا","هذه","ذلك","تلك","هو","هي","هم","هن","انت","أنت","نحن",
    "كان","كانت","يكون","يكونون","الذين","التي","الذي","بينهم","فيما","الي"
])

# ===================== Load data/index/models =====================
_device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading data...")
df_verses = pd.read_excel(QURAN_PATH)
df_maqas  = pd.read_excel(MAQAS_PATH)

print("Loading embeddings/index...")
embeddings = np.load(EMB_PATH).astype("float32")
index = faiss.read_index(INDEX_PATH)

assert len(df_verses) == index.ntotal == embeddings.shape[0], "Mismatch embeddings/index/df_verses!"
print("Verses:", len(df_verses), "| MAQAS rows:", len(df_maqas), "| Index:", index.ntotal)

# Pre-normalize verse text for fast phrase matching
verse_ar_norm = df_verses[AR_NOD].astype(str).map(normalize_ar).tolist()
verse_en_norm = df_verses[EN_COL].astype(str).map(normalize_en).tolist()

# vkey maps
vkey_to_row = {}
row_to_vkey = {}
for i, r in df_verses.iterrows():
    vk = vkey(r[SURA_COL], r[AYAH_COL])
    vkey_to_row[vk] = int(i)
    row_to_vkey[int(i)] = vk

# Build MAQAS inverted index (stems + gloss tokens)
print("Building MAQAS indices...")
df_m = df_maqas.copy()
df_m["__morph"] = df_m[M_TYPE_COL].astype(str).str.lower()
df_m["__stem"]  = df_m[M_NODIAC_COL].astype(str).map(normalize_ar)
df_m["__gloss"] = df_m[M_GLOSS_COL].astype(str).map(normalize_en)

df_stem = df_m[df_m["__morph"].str.contains("stem", na=False)].copy()

verse_ar_tokens: Dict[str, set] = {}
verse_en_tokens: Dict[str, set] = {}

for _, r in df_stem.iterrows():
    vk = vkey(r[M_SURA_COL], r[M_AYAH_COL])
    tok_ar = r["__stem"]
    tok_en = r["__gloss"]

    verse_ar_tokens.setdefault(vk, set())
    verse_en_tokens.setdefault(vk, set())

    if tok_ar:
        verse_ar_tokens[vk].add(tok_ar)

    if tok_en:
        for w in tok_en.split():
            if len(w) >= 4:
                verse_en_tokens[vk].add(w)

inv_ar: Dict[str, set] = {}
inv_en: Dict[str, set] = {}

for vk, toks in verse_ar_tokens.items():
    for t in toks:
        inv_ar.setdefault(t, set()).add(vk)

for vk, toks in verse_en_tokens.items():
    for t in toks:
        inv_en.setdefault(t, set()).add(vk)

print("Arabic stem vocab:", len(inv_ar), "| English gloss vocab:", len(inv_en))

# Models
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
        enc = {k:v.to(_device) for k,v in enc.items()}
        out = mdl_rr(**enc)
        sc = out.logits.squeeze(-1).detach().float().cpu().numpy()
        scores.extend(sc.tolist())
    return np.array(scores, dtype=np.float32)

def build_passage(ix: int) -> str:
    row = df_verses.iloc[int(ix)]
    return f"{row[AR_DIAC]} [SEP] {row[EN_COL]}"

# ===================== MAQAS candidates + phrase hits =====================
def maqas_family_vkeys_ar(base: str) -> set:
    b = normalize_ar(base)
    if not b:
        return set()
    out = set(inv_ar.get(b, set()))
    for tok, vset in inv_ar.items():
        if tok == b or tok.startswith(b) or (b in tok):
            out |= vset
    return out

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
    return [i for i, txt in enumerate(verse_ar_norm) if ph in txt]

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
    return [i for i, txt in enumerate(verse_ar_norm) if needle in f" {txt} "]

def exact_word_hits_en(word_en: str) -> List[int]:
    w = normalize_en(word_en)
    if not w:
        return []
    needle = f" {w} "
    return [i for i, txt in enumerate(verse_en_norm) if needle in f" {txt} "]

# ===================== Expansion mining (Arabic) =====================
def pick_anchor_tokens_ar(query_ar: str, max_tokens=2) -> List[str]:
    toks = [t for t in normalize_ar(query_ar).split() if t and t not in AR_STOP and len(t) >= 3]
    if not toks:
        toks = [t for t in normalize_ar(query_ar).split() if t and len(t) >= 2 and t not in ("ال",)]
    seen=set()
    toks=[t for t in toks if not (t in seen or seen.add(t))]
    return toks[:max_tokens]

def extract_ngrams_containing_anchor(text: str, anchors: List[str], n_min=2, n_max=4) -> List[str]:
    words = [w for w in text.split() if w]
    out = []
    if not words or not anchors:
        return out
    for n in range(n_min, n_max+1):
        for i in range(0, len(words)-n+1):
            ng = words[i:i+n]
            if not any(a in ng for a in anchors):
                continue
            if ng[0] in AR_STOP or ng[-1] in AR_STOP:
                continue
            content = [t for t in ng if (t not in AR_STOP and len(t) >= 3)]
            if len(content) < 2:
                continue
            if "يوم" in anchors and "يوم" in ng:
                j = ng.index("يوم")
                if j+1 < len(ng) and ng[j+1] in AR_STOP:
                    continue
            out.append(" ".join(ng))
    return out

def mine_expansions_ar(query_ar: str, guaranteed_ids: List[int], top_exp=20) -> Tuple[List[str], List[str]]:
    qn = normalize_ar(query_ar)
    anchors = pick_anchor_tokens_ar(query_ar, max_tokens=2)
    if not guaranteed_ids or not anchors:
        return [], anchors
    from collections import Counter
    ref_counts = Counter()
    global_counts = Counter()
    for txt in verse_ar_norm:
        for ph in set(extract_ngrams_containing_anchor(txt, anchors, 2, 4)):
            global_counts[ph] += 1
    for ix in guaranteed_ids:
        txt = verse_ar_norm[int(ix)]
        for ph in set(extract_ngrams_containing_anchor(txt, anchors, 2, 4)):
            ref_counts[ph] += 1
    scored = []
    for ph, c in ref_counts.items():
        if ph == qn:
            continue
        if len(ph) < 5:
            continue
        g = global_counts.get(ph, 0)
        score = c / (g + 1.0)
        if c >= 2 or len(guaranteed_ids) < 25:
            scored.append((ph, score, c, g))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    final = []
    seen = set()
    for ph, _, _, _ in scored:
        if ph in seen:
            continue
        seen.add(ph)
        final.append(ph)
        if len(final) >= top_exp:
            break
    return final, anchors

# ===================== Main Search =====================
def search_api(query: str,
               k_faiss: int = 1200,
               top_expansions: int = 12,
               rerank_limit_non_guaranteed: int = 250,
               rerank_batch: int = 32) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    q = (query or "").strip()
    if not q:
        return [], {"error": "empty"}

    ar_query = is_arabic(q)

    # (1) guaranteed set
    if ar_query:
        maqas_vkeys, toks = maqas_candidates_ar(q)
        qn = normalize_ar(q)
        if " " in qn:
            phrase_ids = exact_phrase_hits_ar(q)
        else:
            phrase_ids = exact_word_hits_ar(q)
    else:
        maqas_vkeys, toks = maqas_candidates_en(q)
        qn = normalize_en(q)
        if " " in qn:
            phrase_ids = exact_phrase_hits_en(q)
        else:
            phrase_ids = exact_word_hits_en(q)

    maqas_ids = []
    for vk in maqas_vkeys:
        if vk in vkey_to_row:
            maqas_ids.append(vkey_to_row[vk])
    maqas_ids = sorted(set(maqas_ids))

    guaranteed_ids = sorted(set(maqas_ids) | set(phrase_ids))
    guaranteed_set = set(guaranteed_ids)

    # (2) expansions (Arabic only)
    expansions, anchors = [], []
    if ar_query:
        expansions, anchors = mine_expansions_ar(q, guaranteed_ids, top_exp=top_expansions)

    # (3) FAISS candidates
    if ar_query and expansions:
        embed_q = normalize_ar(q) + " ; " + " ; ".join(expansions[:8])
    else:
        embed_q = q

    faiss_ids, faiss_scores = faiss_candidate_ids(embed_q, k_retrieve=k_faiss)

    # map faiss score
    id2fs = {int(i): float(s) for i, s in zip(faiss_ids.tolist(), faiss_scores.tolist())}

    # non-guaranteed from faiss (limit)
    other_part = [int(ix) for ix in faiss_ids.tolist() if int(ix) not in guaranteed_set]
    other_part.sort(key=lambda x: id2fs.get(int(x), -1e9), reverse=True)
    other_part = other_part[:rerank_limit_non_guaranteed]

    # expansion phrase hits
    exp_ids = []
    exp_phrase_hits = {}
    if ar_query and expansions:
        for ph in expansions[:top_expansions]:
            hits = exact_phrase_hits_ar(ph)
            if hits:
                exp_phrase_hits[ph] = hits
    if exp_phrase_hits:
        exp_ids = sorted(set([i for lst in exp_phrase_hits.values() for i in lst]))

    # union candidate pool
    union_ids = sorted(set(guaranteed_ids) | set(exp_ids) | set(other_part))
    if not union_ids:
        return [], {"error": "no candidates"}

    # (4) rerank
    if ar_query:
        rr_query = f"أوجد آيات في القرآن تتعلق بمفهوم: {q}. أعد الآيات المرتبطة معنى وسياقًا."
    else:
        rr_query = f"Find Quran verses that discuss the concept of: {q}. Return verses related by meaning and context."

    passages = [build_passage(ix) for ix in union_ids]
    rr_scores = rerank_bge(rr_query, passages, batch_size=rerank_batch, max_length=384)
    rr_map = {int(ix): float(sc) for ix, sc in zip(union_ids, rr_scores)}

    # ===================== Build rows =====================
    q_phrase = normalize_ar(q) if ar_query else normalize_en(q)
    rows = []
    for ix in union_ids:
        row = df_verses.iloc[int(ix)]
        vk = row_to_vkey[int(ix)]

        if ar_query:
            txt_norm = verse_ar_norm[int(ix)]
            if " " in q_phrase:
                is_exact_phrase = int(q_phrase in txt_norm)
            else:
                is_exact_phrase = int(f" {q_phrase} " in f" {txt_norm} ")
        else:
            txt_norm = verse_en_norm[int(ix)]
            if " " in q_phrase:
                is_exact_phrase = int(q_phrase in txt_norm)
            else:
                is_exact_phrase = int(f" {q_phrase} " in f" {txt_norm} ")

        matched_exp = ""
        is_expansion_hit = 0
        if ar_query and expansions:
            for ph in expansions[:top_expansions]:
                phn = normalize_ar(ph)
                if phn and phn in txt_norm:
                    matched_exp = ph
                    is_expansion_hit = 1
                    break

        guaranteed = 1 if int(ix) in guaranteed_set else 0

        priority = 0
        if is_exact_phrase:
            priority = 3
        elif is_expansion_hit:
            priority = 2
        elif guaranteed:
            priority = 1

        rows.append({
            "ix": int(ix),
            "ref": vk,
            "sura": int(row[SURA_COL]),
            "ayah": int(row[AYAH_COL]),
            "score_rr": float(rr_map.get(int(ix), -999.0)),
            "priority": int(priority),
            "exact_phrase": int(is_exact_phrase),
            "guaranteed": int(guaranteed),
            "matched_expansion": matched_exp,
            "arabic": str(row[AR_DIAC]),
            "english": str(row[EN_COL]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["priority", "score_rr"], ascending=[False, False]).reset_index(drop=True)

    # ===================== Final filtering =====================
    df_keep = df[df["priority"] > 0].copy()
    df_sem = df[df["priority"] == 0].copy()
    df_sem = df_sem.sort_values("score_rr", ascending=False)
    TOP_SEM = 150
    RR_MIN = -5.0
    df_sem = df_sem[df_sem["score_rr"] >= RR_MIN].head(TOP_SEM)
    df_final = pd.concat([df_keep, df_sem], ignore_index=True)
    df_final = df_final.sort_values(["priority", "score_rr"], ascending=[False, False]).reset_index(drop=True)
    df_final.insert(0, "rank", np.arange(1, len(df_final) + 1))

    results = df_final[["rank", "ref", "arabic", "english"]].to_dict(orient="records")
    info = {
        "query": q,
        "ar_query": bool(ar_query),
        "total": int(len(df_final)),
    }
    return results, info
