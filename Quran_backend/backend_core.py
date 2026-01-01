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

print("BACKEND_CORE LOADED FROM:", __file__)

# ===================== Device =====================
_device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== Load data/index/models =====================
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

# ===================== Build MAQAS inverted index =====================
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

# ===================== Embedder + Reranker =====================
print("Loading embedder + reranker...")
embedder = SentenceTransformer("BAAI/bge-m3", device=_device)

reranker_name = "BAAI/bge-reranker-v2-m3"
tok_rr = AutoTokenizer.from_pretrained(reranker_name)
mdl_rr = AutoModelForSequenceClassification.from_pretrained(reranker_name).to(_device)
mdl_rr.eval()
print("Ready ✅ on device:", _device)

# ===================== Local LLM Judge (semantic ONLY) =====================
# ملاحظة: LLM هنا "حكم" على النتائج semantic فقط. لا يدخل في lexical نهائياً.
USE_LLM_SEM_JUDGE = os.getenv("USE_LLM_SEM_JUDGE", "1") == "1"

# مناسب لـ T4 16GB (4bit)
LLM_NAME = os.getenv("LLM_NAME", "Qwen/Qwen2.5-7B-Instruct")  # قوي عربي/إنجليزي
LLM_MAX_INPUT_VERSES = int(os.getenv("LLM_MAX_INPUT_VERSES", "60"))  # كم آية نفحصها بالـ LLM
LLM_CONF_MIN = float(os.getenv("LLM_CONF_MIN", "0.60"))  # حد الثقة للقبول
LLM_KEEP_MAYBE = os.getenv("LLM_KEEP_MAYBE", "0") == "1"  # لو تبين maybe يدخل

_llm_tokenizer = None
_llm_model = None
_llm_cache: Dict[str, Dict[str, Any]] = {}  # cache per (query|ref|arabic|english)

def _load_llm_if_needed():
    global _llm_tokenizer, _llm_model
    if _llm_model is not None and _llm_tokenizer is not None:
        return
    if not USE_LLM_SEM_JUDGE:
        return

    print(f"Loading local LLM judge: {LLM_NAME} (4bit) ...")
    _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=True)

    # 4-bit quantization (bitsandbytes)
    _llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
        low_cpu_mem_usage=True
    )
    _llm_model.eval()
    print("LLM judge ready ✅")

def _safe_json_extract(text: str) -> Any:
    """
    يحاول يلقط JSON من الرد حتى لو فيه كلام حوله.
    """
    text = text.strip()
    # أول محاولة: لو النص نفسه JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # محاولة استخراج أول كتلة JSON
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        return None
    chunk = m.group(1).strip()
    try:
        return json.loads(chunk)
    except Exception:
        return None

def _build_llm_prompt(query: str, items: List[Dict[str, str]]) -> str:
    """
    Prompt صارم: يمنع model يسوي expansions/lexical. فقط relevance.
    outputs: JSON array of objects.
    """
    # نختصر النصوص شوي عشان التوكنز
    def cut(s, n):
        s = "" if s is None else str(s)
        s = s.strip()
        return s[:n]

    lines = []
    for it in items:
        lines.append({
            "ref": it["ref"],
            "arabic": cut(it["arabic"], 260),
            "english": cut(it["english"], 260)
        })

    payload = json.dumps(lines, ensure_ascii=False)

    return f"""
You are a strict relevance judge for Quran verse retrieval.

RULES (must follow):
- DO NOT rewrite, paraphrase, expand, or change the query.
- DO NOT generate synonyms or lexical expansions.
- ONLY decide whether each verse is relevant to the query concept by meaning/context.
- Output MUST be valid JSON ONLY. No extra text.

Query: {query}

Decide for each item:
label: one of ["relevant","not_relevant","maybe"]
confidence: number from 0.0 to 1.0
theme: short tag like ["qiyamah","akhirah","hisab","general_reminder","dua","warning","law","story","other"]

INPUT_ITEMS_JSON:
{payload}

Return JSON array of same length, each object:
{{"ref":"<ref>","label":"...","confidence":0.0,"theme":"..."}}
""".strip()

@torch.inference_mode()
def llm_judge_semantic(query: str, sem_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    sem_rows: list of dicts contains ref, arabic, english
    return map ref -> {label, confidence, theme}
    """
    if not USE_LLM_SEM_JUDGE:
        return {}

    _load_llm_if_needed()
    if _llm_model is None:
        return {}

    # caching
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

    # batch them into small chunks to avoid long context
    CHUNK = 12  # آمن على T4
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i+CHUNK]
        items = [{"ref": r["ref"], "arabic": r.get("arabic",""), "english": r.get("english","")} for _, r in chunk]
        prompt = _build_llm_prompt(query, items)

        messages = [{"role":"user","content":prompt}]
        # Qwen instruct uses chat template
        if hasattr(_llm_tokenizer, "apply_chat_template"):
            text = _llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = _llm_tokenizer(text, return_tensors="pt").to(_llm_model.device)
        else:
            inputs = _llm_tokenizer(prompt, return_tensors="pt", truncation=True).to(_llm_model.device)

        gen = _llm_model.generate(
            **inputs,
            max_new_tokens=420,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            eos_token_id=_llm_tokenizer.eos_token_id
        )

        decoded = _llm_tokenizer.decode(gen[0], skip_special_tokens=True)

        # بعض templates يرجع prompt مع output، فنحاول نلقط JSON
        parsed = _safe_json_extract(decoded)

        if not isinstance(parsed, list):
            # fallback: اعتبر الكل maybe بثقة منخفضة (عشان ما نخرب)
            parsed = [{"ref": it["ref"], "label": "maybe", "confidence": 0.45, "theme": "other"} for it in items]

        # build map
        pred_map = {str(x.get("ref","")): x for x in parsed if isinstance(x, dict)}
        for key, r in chunk:
            ref = r["ref"]
            x = pred_map.get(ref, {"ref":ref, "label":"maybe", "confidence":0.45, "theme":"other"})
            rec = {
                "label": str(x.get("label","maybe")),
                "confidence": float(x.get("confidence", 0.45)),
                "theme": str(x.get("theme","other"))
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
        enc = {k:v.to(_device) for k,v in enc.items()}
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
# (كما هو - LLM ممنوع يدخل هنا)
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
        return [], {"error": "empty", "total": 0}

    ar_query = is_arabic(q)

    # (1) guaranteed set (exact word/phrase + MAQAS hits) => LEXICAL (لا تدخل LLM)
    if ar_query:
        maqas_vkeys, toks = maqas_candidates_ar(q)
        qn = normalize_ar(q)
        phrase_ids = exact_phrase_hits_ar(q) if " " in qn else exact_word_hits_ar(q)
    else:
        maqas_vkeys, toks = maqas_candidates_en(q)
        qn = normalize_en(q)
        phrase_ids = exact_phrase_hits_en(q) if " " in qn else exact_word_hits_en(q)

    maqas_ids = [vkey_to_row[vk] for vk in maqas_vkeys if vk in vkey_to_row]
    maqas_ids = sorted(set(maqas_ids))

    guaranteed_ids = sorted(set(maqas_ids) | set(phrase_ids))
    guaranteed_set = set(guaranteed_ids)

    # (2) expansions (Arabic only) - برضه lexical/pseudo-lexical (LLM لا يدخل)
    expansions, anchors = [], []
    if ar_query:
        expansions, anchors = mine_expansions_ar(q, guaranteed_ids, top_exp=top_expansions)

    # (3) FAISS candidates pool
    embed_q = q
    if ar_query and expansions:
        embed_q = normalize_ar(q) + " ; " + " ; ".join(expansions[:8])

    faiss_ids, faiss_scores = faiss_candidate_ids(embed_q, k_retrieve=k_faiss)
    id2fs = {int(i): float(s) for i, s in zip(faiss_ids.tolist(), faiss_scores.tolist())}

    other_part = [int(ix) for ix in faiss_ids.tolist() if int(ix) not in guaranteed_set]
    other_part.sort(key=lambda x: id2fs.get(int(x), -1e9), reverse=True)
    other_part = other_part[:rerank_limit_non_guaranteed]

    # (4) expansion phrase hits
    exp_ids = []
    exp_phrase_hits = {}
    if ar_query and expansions:
        for ph in expansions[:top_expansions]:
            hits = exact_phrase_hits_ar(ph)
            if hits:
                exp_phrase_hits[ph] = hits
    if exp_phrase_hits:
        exp_ids = sorted(set([i for lst in exp_phrase_hits.values() for i in lst]))

    # Final candidate pool
    union_ids = sorted(set(guaranteed_ids) | set(exp_ids) | set(other_part))
    if not union_ids:
        return [], {"error": "no candidates", "total": 0}

    # (5) reranker query
    if ar_query:
        rr_query = f"أوجد آيات في القرآن تتعلق بمفهوم: {q}. أعد الآيات المرتبطة معنى وسياقًا."
    else:
        rr_query = f"Find Quran verses that discuss the concept of: {q}. Return verses related by meaning and context."

    passages = [build_passage(ix) for ix in union_ids]
    rr_scores = rerank_bge(rr_query, passages, batch_size=rerank_batch, max_length=384)
    rr_map = {int(ix): float(sc) for ix, sc in zip(union_ids, rr_scores)}

    # (6) build rows + priority layers
    q_phrase = normalize_ar(q) if ar_query else normalize_en(q)

    rows = []
    for ix in union_ids:
        row = df_verses.iloc[int(ix)]
        vk = row_to_vkey[int(ix)]

        txt_norm = verse_ar_norm[int(ix)] if ar_query else verse_en_norm[int(ix)]

        # exact word vs exact phrase
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

        # priority: 3 exact, 2 expansion, 1 guaranteed, 0 semantic
        priority = 0
        if is_exact_phrase:
            priority = 3
        elif is_expansion_hit:
            priority = 2
        elif guaranteed:
            priority = 1

        bucket = "lexical" if priority > 0 else "semantic"

        rows.append({
            "ix": int(ix),
            "ref": vk,
            "score_rr": float(rr_map.get(int(ix), -999.0)),
            "priority": int(priority),
            "bucket": bucket,
            "arabic": str(row[AR_DIAC]),
            "english": str(row[EN_COL]),
            "matched_expansion": matched_exp
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["priority", "score_rr"], ascending=[False, False]).reset_index(drop=True)

    # ✅ (A) LEXICAL KEEP (NO LLM)
    df_keep = df[df["priority"] > 0].copy()

    # ✅ (B) SEMANTIC candidates
    df_sem = df[df["priority"] == 0].copy()
    df_sem = df_sem.sort_values("score_rr", ascending=False)

    # بدل TOP_SEM/RR_MIN الثابتة => نستخدم LLM (semantic only)
    sem_candidates = df_sem.head(max(LLM_MAX_INPUT_VERSES, 1)).to_dict(orient="records")

    sem_keep_refs = set()
    sem_meta = {}

    if USE_LLM_SEM_JUDGE and len(sem_candidates) > 0:
        preds = llm_judge_semantic(q, sem_candidates)  # map ref -> {label, confidence, theme}
        for r in sem_candidates:
            ref = r["ref"]
            p = preds.get(ref, None)
            if not p:
                continue
            lab = p.get("label", "maybe")
            conf = float(p.get("confidence", 0.45))
            theme = p.get("theme", "other")

            ok = False
            if lab == "relevant" and conf >= LLM_CONF_MIN:
                ok = True
            elif LLM_KEEP_MAYBE and lab == "maybe" and conf >= (LLM_CONF_MIN + 0.10):
                ok = True

            if ok:
                sem_keep_refs.add(ref)
                sem_meta[ref] = {"llm_label": lab, "llm_conf": conf, "llm_theme": theme}

    # fallback بسيط لو LLM مقفل/فشل: احتفظ بأفضل كم آية حسب reranker
    if not USE_LLM_SEM_JUDGE:
        # نفس منطقك القديم لكن بدون رقم ثابت كبير
        RR_MIN = -5.0
        TOP_SEM = 150
        df_sem_keep = df_sem[df_sem["score_rr"] >= RR_MIN].head(TOP_SEM).copy()
    else:
        # فلترة حسب LLM
        if len(sem_keep_refs) == 0:
            # لو LLM رفض كل شيء (نادر)، نعطي قليل جداً كاحتياط (اختياري)
            # ممكن تخليه 0 لو تبين strict 100%
            df_sem_keep = df_sem.head(10).copy()
            df_sem_keep["llm_label"] = "fallback"
            df_sem_keep["llm_conf"] = 0.40
            df_sem_keep["llm_theme"] = "other"
        else:
            df_sem_keep = df_sem[df_sem["ref"].isin(list(sem_keep_refs))].copy()
            df_sem_keep["llm_label"] = df_sem_keep["ref"].map(lambda x: sem_meta.get(x, {}).get("llm_label", ""))
            df_sem_keep["llm_conf"]  = df_sem_keep["ref"].map(lambda x: sem_meta.get(x, {}).get("llm_conf", 0.0))
            df_sem_keep["llm_theme"] = df_sem_keep["ref"].map(lambda x: sem_meta.get(x, {}).get("llm_theme", ""))

            # نرتب semantic: أولاً ثقة LLM ثم rr_score
            df_sem_keep = df_sem_keep.sort_values(["llm_conf", "score_rr"], ascending=[False, False])

    # ✅ Final merge
    df_final = pd.concat([df_keep, df_sem_keep], ignore_index=True)
    df_final = df_final.sort_values(["priority", "score_rr"], ascending=[False, False]).reset_index(drop=True)
    df_final.insert(0, "rank", np.arange(1, len(df_final) + 1))

    # output (تقدرين تضيفين llm columns لو تبين في الـ API)
    keep_cols = ["rank", "ref", "bucket", "arabic", "english"]
    if USE_LLM_SEM_JUDGE:
        # اختياري لإظهار سبب قبول semantic
        if "llm_label" in df_final.columns:
            keep_cols += ["llm_label", "llm_conf", "llm_theme"]

    results = df_final[keep_cols].to_dict(orient="records")

    info = {
        "query": q,
        "ar_query": bool(ar_query),
        "total": int(len(df_final)),
        "lexical_total": int(len(df_keep)),
        "semantic_total": int(len(df_sem_keep)),
        "llm_sem_judge": bool(USE_LLM_SEM_JUDGE),
        "llm_name": LLM_NAME if USE_LLM_SEM_JUDGE else None
    }
    return results, info
