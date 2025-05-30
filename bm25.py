import requests
import json
import math

# TREC topics URL (misol uchun)
url = "https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.topics.json"

# Ma'lumotlarni yuklab olish
response = requests.get(url)
topics = json.loads(response.text)

# Kichik qismini ko'rib chiqamiz
print("Misol uchun birinchi so'rov:")
print(topics[0])

# Oddiy dokumentlar bazasi (TREC hujjatlari emas, faqat misol uchun)
documents = {
    "D1": "The economy is growing rapidly with new policies.",
    "D2": "Inflation impacts the economy in many ways.",
    "D3": "Fiscal policies guide the economic growth.",
    "D4": "Monetary policy stabilizes the economy."
}

# BM25 parametrlar
k1 = 1.5
b = 0.75

N = len(documents)

def tokenize(text):
    return text.lower().split()

doc_lens = {doc_id: len(tokenize(text)) for doc_id, text in documents.items()}
avgdl = sum(doc_lens.values()) / N

def compute_tf(doc_tokens):
    tf = {}
    for term in doc_tokens:
        tf[term] = tf.get(term, 0) + 1
    return tf

def compute_df(documents_tokens):
    df = {}
    for tokens in documents_tokens.values():
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1
    return df

documents_tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}

tf_docs = {doc_id: compute_tf(tokens) for doc_id, tokens in documents_tokens.items()}
df = compute_df(documents_tokens)

def compute_idf(term):
    n_qi = df.get(term, 0)
    return math.log((N - n_qi + 0.5) / (n_qi + 0.5) + 1)

def bm25_score(doc_id, query_terms):
    score = 0.0
    tf = tf_docs[doc_id]
    dl = doc_lens[doc_id]
    for term in query_terms:
        if term not in tf:
            continue
        idf = compute_idf(term)
        freq = tf[term]
        numerator = freq * (k1 + 1)
        denominator = freq + k1 * (1 - b + b * dl / avgdl)
        score += idf * (numerator / denominator)
    return score

# TREC so'rovlaridan birinchi so'rovni tanlab olamiz (misol)
query = topics[0]["title"]
query_terms = tokenize(query)

scores = {}
for doc_id in documents:
    scores[doc_id] = bm25_score(doc_id, query_terms)

sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print(f"So'rov: {query}")
print("BM25 bo'yicha hujjatlar reytingi:")
for doc_id, score in sorted_scores:
    print(f"{doc_id}: {score:.4f}")




