# backend_core.py
import os
import re
import json
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

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

# ===================== Local LLM Judge (semantic ONLY) =====================
USE_LLM_SEM_JUDGE = (os.getenv("LLM_FILTER_ENABLED") or os.getenv("USE_LLM_SEM_JUDGE") or "1") == "1"

LLM_NAME = (
    os.getenv("LLM_MODEL_NAME")
    or os.getenv("LLM_NAME")
    or "Qwen/Qwen2.5-1.5B-Instruct"
)

LLM_BATCH_SIZE      = int(os.getenv("LLM_BATCH_SIZE", "12"))
LLM_MAX_NEW_TOKENS   = int(os.getenv("LLM_MAX_NEW_TOKENS", "64"))
LLM_TEXT_TRIM        = int(os.getenv("LLM_TEXT_TRIM", "280"))
LLM_MAX_INPUT_VERSES = int(os.getenv("LLM_MAX_INPUT_VERSES", "50"))  # يدخل للـ LLM
LLM_CONF_MIN         = float(os.getenv("LLM_CONF_MIN", "0.45"))
LLM_KEEP_MAYBE       = os.getenv("LLM_KEEP_MAYBE", "0") == "1"

_llm_tokenizer = None
_llm_model = None
_llm_cache: Dict[str, Dict[str, Any]] = {}

def _load_llm_if_needed():
    global _llm_tokenizer, _llm_model
    if not USE_LLM_SEM_JUDGE:
        return
    if _llm_model is not None and _llm_tokenizer is not None:
        return

    print(f"Loading local LLM judge: {LLM_NAME} (4bit) ...")
    _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=True)

    _llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
        low_cpu_mem_usage=True,
    )
    _llm_model.eval()
    print("LLM judge ready ✅")

def _safe_json_extract(text: str) -> Any:
    text = (text or "").strip()

    # remove common markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # first try direct
    try:
        return json.loads(text)
    except Exception:
        pass

    # try to find the FIRST json array non-greedily
    m = re.search(r"\[\s*[\s\S]*?\s*\]", text)
    if m:
        chunk = m.group(0).strip()
        try:
            return json.loads(chunk)
        except Exception:
            return None

    # fallback: try first json object
    m = re.search(r"\{\s*[\s\S]*?\s*\}", text)
    if m:
        chunk = m.group(0).strip()
        try:
            return json.loads(chunk)
        except Exception:
            return None

    return None


def _cut(s: str, n: int) -> str:
    s = "" if s is None else str(s).strip()
    return s[:n]

def _build_llm_prompt_batch(query: str, items: List[Dict[str, str]]) -> str:
    lines = []
    for it in items:
        lines.append({
            "ref": it["ref"],
            "arabic": _cut(it.get("arabic",""), LLM_TEXT_TRIM),
            "english": _cut(it.get("english",""), LLM_TEXT_TRIM),
        })

    payload = json.dumps(lines, ensure_ascii=False)

    return f"""
You are a strict relevance judge for Quran verse retrieval.

ABSOLUTE OUTPUT FORMAT (critical):
- Your entire output MUST be a single JSON array.
- Output MUST start with '[' and end with ']'.
- Do NOT wrap in markdown (no ```json).
- Do NOT add any text before or after the JSON.
- If you output anything else, your output will be discarded.

TASK:
- Given the query, decide if each verse is relevant by meaning/context (NOT keyword match).
- Do NOT rewrite or expand the query.
- Do NOT generate synonyms.

LABELS:
- "relevant": clearly about the query concept by meaning/context.
- "not_relevant": not about the concept.
- "maybe": partially related or unclear.

CONFIDENCE:
- A number from 0.0 to 1.0.

THEME:
- Choose ONE tag from:
  ["qiyamah","akhirah","hisab","general_reminder","dua","warning","law","story","other"]
- If unsure, use "other". (Do not invent new tags.)

EXAMPLE OUTPUT (format only):
[
  {{"ref":"2:25","label":"relevant","confidence":0.78,"theme":"akhirah"}},
  {{"ref":"2:26","label":"not_relevant","confidence":0.74,"theme":"other"}}
]

Query: {query}

INPUT_ITEMS_JSON:
{payload}

Return ONLY the JSON array of the same length as INPUT_ITEMS_JSON.
Each object MUST be:
{{"ref":"<ref>","label":"relevant|not_relevant|maybe","confidence":0.0,"theme":"<tag>"}}
""".strip()

@torch.inference_mode()
def llm_judge_semantic(query: str, sem_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not USE_LLM_SEM_JUDGE:
        return {}

    _load_llm_if_needed()
    if _llm_model is None:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    todo = []

    for r in sem_rows:
        key = f"{query}||{r['ref']}||{r.get('arabic','')}||{r.get('english','')}"
        if key in _llm_cache:
            out[r["ref"]] = _llm_cache[key]
        else:
            todo.append((key, r))

    if not todo:
        return out

    CHUNK = max(1, LLM_BATCH_SIZE)
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i+CHUNK]
        items = [{"ref": r["ref"], "arabic": r.get("arabic",""), "english": r.get("english","")} for _, r in chunk]
        prompt = _build_llm_prompt_batch(query, items)

        messages = [{"role": "user", "content": prompt}]
        if hasattr(_llm_tokenizer, "apply_chat_template"):
            text = _llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = _llm_tokenizer(text, return_tensors="pt").to(_llm_model.device)
        else:
            inputs = _llm_tokenizer(prompt, return_tensors="pt", truncation=True).to(_llm_model.device)

        gen = _llm_model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            eos_token_id=_llm_tokenizer.eos_token_id,
        )

        in_len = inputs["input_ids"].shape[1]
        new_tokens = gen[0][in_len:]
        decoded = _llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    parsed = _safe_json_extract(decoded)

    # if parsing failed or not a list => fallback
    if not isinstance(parsed, list):
        parsed = [{"ref": it["ref"], "label": "maybe", "confidence": 0.45, "theme": "other"} for it in items]

    # normalize refs in model output
    norm_list = []
    for x in parsed:
        if isinstance(x, dict):
            x_ref = str(x.get("ref", "")).strip()
            x["ref"] = x_ref
            norm_list.append(x)

    pred_map = {x.get("ref",""): x for x in norm_list if x.get("ref","")}

    # EXTRA fallback: if pred_map is empty or refs don't match, map by order
    use_order_fallback = (len(pred_map) == 0)

    for i, (key, r) in enumerate(chunk):
        ref = str(r["ref"]).strip()

        if use_order_fallback and i < len(norm_list):
            x = norm_list[i]
        else:
            x = pred_map.get(ref)

        if not isinstance(x, dict):
            x = {"ref": ref, "label": "maybe", "confidence": 0.45, "theme": "other"}

        rec = {
            "label": str(x.get("label", "maybe")).strip().lower(),
            "confidence": float(x.get("confidence", 0.45)),
            "theme": str(x.get("theme", "other")).strip(),
        }
        _llm_cache[key] = rec
        out[ref] = rec

    return out

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

    # (2) FAISS candidates pool (semantic candidates)
    embed_q = q
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

    sem_candidates = df_sem.head(max(LLM_MAX_INPUT_VERSES, 1)).to_dict(orient="records")
    _log(f"[SEM] sem_pool_after_rrmin={len(df_sem)} | llm_input={len(sem_candidates)} | use_llm={USE_LLM_SEM_JUDGE}")

    sem_keep_refs = set()
    sem_meta = {}

    if USE_LLM_SEM_JUDGE and len(sem_candidates) > 0:
        preds = llm_judge_semantic(q, sem_candidates)

        for r in sem_candidates:
            ref = r["ref"]
            p = preds.get(ref)
            if not p:
                continue

            lab = p.get("label", "maybe")
            conf = float(p.get("confidence", 0.45))
            theme = p.get("theme", "other")

            ok = False
            if lab == "relevant" and conf >= LLM_CONF_MIN:
                ok = True
            elif LLM_KEEP_MAYBE and lab == "maybe" and conf >= LLM_CONF_MIN:
                ok = True

            if ok:
                sem_keep_refs.add(ref)
                sem_meta[ref] = {"llm_label": lab, "llm_conf": conf, "llm_theme": theme}

        _log(f"[LLM] judged={len(sem_candidates)} | kept={len(sem_keep_refs)} | conf_min={LLM_CONF_MIN}")

    SEM_RETURN_TOPN = int(os.getenv("SEM_RETURN_TOPN", "20"))

    # (6) build df_sem_keep
    if not USE_LLM_SEM_JUDGE:
        df_sem_keep = df_sem.head(SEM_RETURN_TOPN).copy()
    else:
        if len(sem_keep_refs) == 0:
            df_sem_keep = df_sem.head(0).copy()
        else:
            df_sem_keep = df_sem[df_sem["ref"].isin(list(sem_keep_refs))].copy()
            df_sem_keep["llm_label"] = df_sem_keep["ref"].map(lambda x: sem_meta.get(x, {}).get("llm_label", ""))
            df_sem_keep["llm_conf"]  = df_sem_keep["ref"].map(lambda x: sem_meta.get(x, {}).get("llm_conf", 0.0))
            df_sem_keep["llm_theme"] = df_sem_keep["ref"].map(lambda x: sem_meta.get(x, {}).get("llm_theme", "other"))
            df_sem_keep = df_sem_keep.sort_values(
                ["llm_conf", "score_rr"], ascending=[False, False]
            )

    # ===== Fallback fill to always return TOP-10 semantic =====

    # قص اللي نجح من LLM
    df_sem_keep = df_sem_keep.head(SEM_RETURN_TOPN).copy()

    need = SEM_RETURN_TOPN - len(df_sem_keep)
    if need > 0:
        existing = set(df_sem_keep["ref"].astype(str))
        df_fill = df_sem[~df_sem["ref"].astype(str).isin(existing)].head(need).copy()

        # تعليم fallback بوضوح
        df_fill["llm_label"] = "fallback"
        df_fill["llm_conf"]  = 0.0
        df_fill["llm_theme"] = "fallback"

        df_sem_keep = pd.concat([df_sem_keep, df_fill], ignore_index=True)

        print(f"[SEM-FILL] kept_from_llm={SEM_RETURN_TOPN-need} | filled={need} | final_sem={len(df_sem_keep)}")

    df_final = pd.concat([df_keep, df_sem_keep], ignore_index=True)
    df_final = df_final.sort_values(
        ["priority", "score_rr"], ascending=[False, False]
    ).reset_index(drop=True)

    df_final.insert(0, "rank", np.arange(1, len(df_final) + 1))



    # ensure LLM cols exist if enabled (NO KeyError forever)
    if USE_LLM_SEM_JUDGE:
        if "llm_label" not in df_final.columns:
            df_final["llm_label"] = ""
        if "llm_conf" not in df_final.columns:
            df_final["llm_conf"] = 0.0
        if "llm_theme" not in df_final.columns:
            df_final["llm_theme"] = "other"

    keep_cols = ["rank", "ref", "bucket", "arabic", "english"]
    if USE_LLM_SEM_JUDGE:
        keep_cols += ["llm_label", "llm_conf", "llm_theme"]

    results = df_final.reindex(columns=keep_cols, fill_value="").to_dict(orient="records")

    _log(f"[FINAL] total={len(df_final)} | lexical_total={len(df_keep)} | semantic_total={len(df_sem_keep)}")

    info = {
        "query": q,
        "ar_query": bool(ar_query),
        "total": int(len(df_final)),
        "lexical_total": int(len(df_keep)),
        "semantic_total": int(len(df_sem_keep)),
        "semantic_pool": int(len(df_sem)),
        "llm_sem_judge": bool(USE_LLM_SEM_JUDGE),
        "llm_name": LLM_NAME if USE_LLM_SEM_JUDGE else None,
        "top_sem_total": int(TOP_SEM_TOTAL),
        "llm_max_input_verses": int(LLM_MAX_INPUT_VERSES),
    }
    return results, info
