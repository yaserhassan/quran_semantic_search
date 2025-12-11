import pandas as pd
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer

pd.set_option("display.max_colwidth", 200)
print("Environment ready.")

#--- data preparation ---


QURAN_PATH = "Quran_data.xlsx"
MAQAS_PATH = "MAQAS.xlsx"

df_verses = pd.read_excel(QURAN_PATH)
df_maqas = pd.read_excel(MAQAS_PATH)

AR_COL_WITH = "Quran with diacritic"
AR_COL   = "Quran without diacritic"
EN_COL   = "Translation in English"
SURA_COL = "Chapter"
AYAH_COL = "No verse in Chapter"

M_SURA_COL   = "Sura_No"
M_AYAH_COL   = "Verse_No"
M_WORD_COL   = "Word"
M_NODIAC_COL = "Without_Diacritics"
M_GLOSS_COL  = "Gloss"
M_TYPE_COL   = "Morph_Type"

#----------- prepere text for embedding ----

def make_text(row):
    ar = str(row[AR_COL]).strip()
    en = str(row[EN_COL]).strip()
    return f"{ar} [SEP] {en}"

df_verses["text_for_embedding"] = df_verses.apply(make_text, axis=1)

#--------- load the model and embeddings ----

MODEL_NAME = "BAAI/bge-m3"
model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded:", MODEL_NAME)

EMB_PATH = "embeddings_bge_m3.npy"
INDEX_PATH = "faiss_index_bge_m3.idx"

# Load precomputed embeddings
embeddings = np.load(EMB_PATH)
dim = embeddings.shape[1]

# Load FAISS index
index = faiss.read_index(INDEX_PATH)
print("Index loaded. Vectors:", index.ntotal)
#---------------------------

# helper function to clean and prepare query text

def is_arabic(text: str) -> bool:
    text = str(text) if text is not None else ""
    return re.search(r'[\u0600-\u06FF]', text) is not None

# -- expand query terms based on Arabic and English forms
def expand_query(q: str, max_terms: int = 40):
    q = str(q).strip()
    if not q:
        return [q]

    ar = is_arabic(q)

    nodiac = df_maqas[M_NODIAC_COL].astype(str)
    gloss  = df_maqas[M_GLOSS_COL].astype(str)

    # Search in Arabic or English
    if ar:
        mask = nodiac.str.contains(q, case=False, regex=False)
    else:
        mask = gloss.str.contains(q, case=False, na=False)

    sub = df_maqas[mask]
    extra_terms = []

    # 1) Arabic forms
    extra_terms.extend(sub[M_NODIAC_COL].astype(str).tolist())

    # 2) English forms from Gloss
    gloss_terms = []
    for g in sub[M_GLOSS_COL].astype(str):
        parts = re.split(r'[^a-zA-Z]+', g.lower())
        gloss_terms.extend([p for p in parts if len(p) > 3])
    extra_terms.extend(gloss_terms)

    if not extra_terms:
        return [q]

    # Unique + limit
    all_terms = []
    for t in [q] + extra_terms:
        t = t.strip()
        if t and t not in all_terms:
            all_terms.append(t)
        if len(all_terms) >= max_terms:
            break

    return all_terms


def encode_expanded_query(query: str):
    terms = expand_query(query)
    expanded_text = " ; ".join(terms)

    q_emb = model.encode(
        expanded_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    if q_emb.ndim == 1:
        q_emb = q_emb[None, :]

    return q_emb, terms

#---------------------------

# search function


def quick_search(query: str, top_k: int = 30):

    q_emb, terms = encode_expanded_query(query)
    n = index.ntotal
    k = min(top_k, n)
    scores, ids = index.search(q_emb, k)

    results = []
    for i, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        row = df_verses.iloc[int(idx)]
        results.append({
            "rank": i,
            "score": float(score),
            "sura": int(row[SURA_COL]),
            "ayah": int(row[AYAH_COL]),
            "ref": f"{int(row[SURA_COL])}:{int(row[AYAH_COL])}",
            "arabic": row[AR_COL_WITH],
            "english": row[EN_COL],
        })

    return results


def search_topk_semantic(q: str, top_k: int = 30):
    q_emb, terms = encode_expanded_query(q)

    n = index.ntotal
    k = min(top_k, n)
    scores, idxs = index.search(q_emb, k)
    scores, idxs = scores[0], idxs[0]

    rows = []
    for sc, ix in zip(scores, idxs):
        row = df_verses.iloc[int(ix)]
        rows.append({
            "score_cosine": float(sc),
            "ref": f"{int(row[SURA_COL])}:{int(row[AYAH_COL])}",
            "arabic": row[AR_COL_WITH],
            "english": row[EN_COL],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values("score_cosine", ascending=False).reset_index(drop=True)


def semantic_search(query: str, top_k: int = 30):
    return quick_search(query, top_k=top_k)