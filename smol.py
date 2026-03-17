import os
os.environ["OMP_NUM_THREADS"] = "1"


import pickle
import faiss
import time

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
nltk_stop_words = set(nltk.corpus.stopwords.words('english'))


import sys
from text_utils import tokenize_lemma

# Fix pickle reference
sys.modules['__main__'].tokenize_lemma = tokenize_lemma

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load TF-IDF matrix
with open("tfidf_matrix.pkl", "rb") as f:
    X_train_vectors = pickle.load(f)

# Load questions
with open("questions.pkl", "rb") as f:
    raw_questions = pickle.load(f)

# Load answers
with open("answers.pkl", "rb") as f:
    answers = pickle.load(f)

# Load FAISS index
index = faiss.read_index("faiss_index.index")

print("System loaded successfully!")






from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading LLM...")

LLM_MODEL = "HuggingFaceTB/SmolLM-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

model = model.to("cpu")

print("LLM loaded successfully!")


from sentence_transformers import SentenceTransformer, CrossEncoder

print("Loading embedding models...")

embedder = SentenceTransformer("intfloat/e5-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

print("Embedding models loaded")


FAISS_SIMILARITY_THRESHOLD = 0.40

def classify_question(user_question):
    q_emb = embedder.encode(
        ["query: " + user_question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, _ = index.search(q_emb, 1)
    top_score = float(scores[0][0])

    if top_score >= FAISS_SIMILARITY_THRESHOLD:
        return "DATASET", top_score
    return "GENERAL", top_score

import re

ACTIONS = ["add", "create", "delete", "remove", "update", "edit"]

def split_intents(question):

    q = question.lower().strip()

    # remove filler
    q = re.sub(r'how to|can you|please|help me', '', q)

    # ✅ CASE 1: action-based split (add/delete)
    actions = re.findall(r'\b(' + '|'.join(ACTIONS) + r')\b', q)

    if len(actions) > 1:
        obj = re.sub(r'\b(' + '|'.join(ACTIONS) + r')\b', '', q)
        obj = re.sub(r'and|,|&|then', '', obj).strip()
        return [f"{a} {obj}".strip() for a in actions]

    # ✅ CASE 2: definition-based split (what is X and Y)
    if " and " in q:
        parts = q.split(" and ")

        intents = []
        for p in parts:
            p = p.strip()

            # normalize
            if not p.startswith("what is"):
                p = "what is " + p

            intents.append(p)

        print("Split intents:", intents)
        return intents

    # ✅ default
    return [q]


def retrieve_context(user_question):

    q_emb = embedder.encode(
        ["query: " + user_question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # 🔍 Search top 10
    scores, idxs = index.search(q_emb, 10)

    candidates = []

    for s, i in zip(scores[0], idxs[0]):
        candidates.append({
            "question": raw_questions[int(i)],
            "answer": answers[int(i)],
            "faiss_score": float(s)
        })

    # 🔥 RERANK (VERY IMPORTANT)
    pairs = [
        (user_question, f"{c['question']} {c['answer']}")
        for c in candidates
    ]

    rerank_scores = reranker.predict(pairs)

    for c, rs in zip(candidates, rerank_scores):
        c["rerank_score"] = float(rs)

    # sort best first
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    best = candidates[0]

    print("Best match:", best["question"])
    print("FAISS score:", best["faiss_score"])
    print("RERANK score:", best["rerank_score"])

    # ✅ KEY FIX (same as Colab)
    if best["rerank_score"] >= 0.55:
        return best["answer"], best["rerank_score"]

    return None, best["rerank_score"]

def build_prompt(system_prompt, user_prompt):

    return f"""System:
{system_prompt}

User:
{user_prompt}

Assistant:
"""

def rag_predict(context, question):

    system_prompt = """
You are a Salesforce assistant.
Return the exact answer from the dataset context.
Do not summarize.
Do not shorten.
Return the full answer exactly as provided.
"""

    user_prompt = f"""
Context:
{context}

Question:
{question}
"""

    prompt = build_prompt(system_prompt, user_prompt)

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated.split("Assistant:")[-1].strip()

def general_llm_answer(question):

    prompt = f"""<|system|>
You are a helpful assistant. Always give a complete answer.

<|user|>
{question}

<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return text.split("<|assistant|>")[-1].strip() or "I couldn't generate a response."



import numpy as np

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def clean_intent(intent):
    return intent.replace("  ", " ").strip()

from fastapi import FastAPI

app = FastAPI()


@app.post("/ask")
def ask(question: str):

    print("\nIncoming Question:", question)

    # ✅ STEP 1: Split intents
    intents = split_intents(question)

    final_answers = []

    for intent in intents:

        # ✅ STEP 2: classify (dataset vs general)
        label, score = classify_question(intent)

        print(f"\nIntent: {intent}")
        print(f"Route: {label} (score: {score})")

        # ✅ STEP 3: route logic
        if label == "GENERAL":
            answer = general_llm_answer(intent)

        else:
            context, rerank_score = retrieve_context(intent)

            if context:
                answer = context
            else:
                answer = general_llm_answer(intent)

        final_answers.append(answer)

    # ✅ STEP 4: remove duplicates
    final_output = "\n\n".join(dict.fromkeys(final_answers))

    return {
        "question": question,
        "intents": intents,
        "answer": final_output
    }