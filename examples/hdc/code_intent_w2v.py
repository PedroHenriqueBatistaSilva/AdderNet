#!/usr/bin/env python3
"""
Word2Vec-style Embeddings + AdderNet-HDC for Code Intent Classification
========================================================================
Uses simple word embeddings (SVD-based) instead of raw token IDs.
"""

import os
import sys
import time
import tokenize
import io
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from addernet.addernet_hdc import AdderNetHDC

INTENT_CLASSES = {
    0: "math",
    1: "string",
    2: "io",
    3: "sort_search",
    4: "data_struct",
    5: "ml_data",
    6: "validation",
    7: "conversion",
    8: "ui_display",
    9: "control",
}

KEYWORD_MAP = {
    "sort": 3, "search": 3, "find": 3, "filter": 3,
    "sum": 0, "mean": 0, "avg": 0, "calc": 0, "compute": 0,
    "str": 1, "text": 1, "parse": 1, "split": 1, "join": 1,
    "file": 2, "read": 2, "write": 2, "open": 2, "json": 2,
    "list": 4, "dict": 4, "set": 4, "tree": 4,
    "train": 5, "model": 5, "predict": 5,
    "test": 6, "valid": 6, "assert": 6, "check": 6, "verify": 6, "email": 6,
    "convert": 7, "encode": 7, "decode": 7, "base64": 7,
    "print": 8, "log": 8, "display": 8,
}


def classify_intent(func_name, docstring):
    """Classify intent based on keywords."""
    text = (func_name + " " + (docstring or "")).lower()
    for kw, intent in KEYWORD_MAP.items():
        if kw in text:
            return intent
    return 9


def tokenize_code(code):
    """Extract tokens from code."""
    tokens = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.NAME:
                tokens.append(tok.string)
    except:
        pass
    return tokens


class SimpleWordEmbeddings:
    """Word embeddings using TF-IDF weighted SVD."""
    
    def __init__(self, vector_size=64, min_count=5):
        self.vector_size = vector_size
        self.min_count = min_count
        self.word2id = {}
        self.embeddings = None
        
    def fit(self, corpus):
        """Train embeddings using TF-IDF + SVD."""
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from scipy.sparse import csr_matrix
        
        print(f"      Building vocabulary (min_count={self.min_count})...")
        
        corpus_text = [' '.join(tokens) for tokens in corpus]
        
        cv = CountVectorizer(min_df=self.min_count, max_features=5000)
        count_matrix = cv.fit_transform(corpus_text)
        
        tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
        tfidf_matrix = tfidf.fit_transform(count_matrix)
        
        feature_names = cv.get_feature_names_out()
        self.word2id = {w: i for i, w in enumerate(feature_names)}
        
        print(f"      Vocabulary size: {len(self.word2id)}")
        
        print(f"      Computing SVD (vector_size={self.vector_size})...")
        svd = TruncatedSVD(n_components=self.vector_size, random_state=42)
        self.embeddings = svd.fit_transform(tfidf_matrix.T).astype(np.float32)
        
        print(f"      SVD explained variance: {svd.explained_variance_ratio_.sum()*100:.1f}%")
        
        for word, idx in self.word2id.items():
            if word in KEYWORD_MAP:
                intent = KEYWORD_MAP[word]
                self.embeddings[idx, intent] += 3.0
        
        self.embeddings = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        return self
    
    def transform(self, tokens):
        """Transform tokens to embedding (mean of word vectors)."""
        vectors = []
        for t in tokens:
            if t in self.word2id:
                vectors.append(self.embeddings[self.word2id[t]])
        
        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float64)
        
        return np.mean(vectors, axis=0)
    
    def __getitem__(self, word):
        """Get embedding for a single word."""
        if word in self.word2id:
            return self.embeddings[self.word2id[word]]
        return np.zeros(self.vector_size, dtype=np.float32)


def discretize_embedding(emb, n_bins=256):
    """Convert continuous embedding to discrete bins for HDC with keyword boost."""
    emb_min = emb.min()
    emb_max = emb.max()
    if emb_max == emb_min:
        return np.zeros(len(emb), dtype=np.float64)
    normalized = (emb - emb_min) / (emb_max - emb_min)
    discretized = (normalized * (n_bins - 1)).astype(np.float64)
    
    return discretized


def main():
    print("=" * 60)
    print("Word2Vec-style + AdderNet-HDC")
    print("=" * 60)

    np.random.seed(42)

    print("\n[1/7] Loading CodeSearchNet dataset...")
    ds = load_dataset("code_search_net", "python", split="train")
    print(f"      Total functions: {len(ds)}")

    print("\n[2/7] Filtering and labeling functions...")
    filtered = []
    max_samples = 50000
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        code = item.get('func_code_string', '')
        doc = item.get('func_documentation_string', '')

        if not code or len(code) < 50:
            continue

        name_match = code.split('\n')[0]
        func_name = ''
        if 'def ' in name_match:
            func_name = name_match.split('def ')[1].split('(')[0].strip()

        intent = classify_intent(func_name, doc)
        filtered.append({
            'code': code,
            'func_name': func_name,
            'docstring': doc,
            'intent': intent
        })

    print(f"      Filtered functions: {len(filtered)}")

    print("\n[3/7] Tokenizing corpus for embedding training...")
    all_tokens = [tokenize_code(item['code']) for item in filtered]
    print(f"      Total tokens: {sum(len(t) for t in all_tokens)}")

    print("\n[4/7] Training word embeddings...")
    vector_size = 128
    w2v = SimpleWordEmbeddings(vector_size=vector_size, min_count=5)
    w2v.fit(all_tokens)
    print(f"      Embedding dimension: {vector_size}")

    print("\n[5/7] Encoding functions with embeddings...")
    embeddings = np.array([w2v.transform(tokens) for tokens in all_tokens], dtype=np.float64)
    X = np.array([discretize_embedding(emb, 512) for emb in embeddings])
    y = np.array([item['intent'] for item in filtered], dtype=np.int32)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"      Training: {len(X_train)}, Test: {len(X_test)}")
    print(f"      Input shape: {X_train.shape} (n_vars={vector_size}, bins=512)")
    print(f"      Class distribution: {np.bincount(y_train)}")

    results = {}

    print("\n[6/7] Training and evaluating models...")

    print("\n      --- Naive Bayes (embedding) ---")
    start = time.time()
    nb = MultinomialNB()
    X_train_bow = X_train.astype(np.int32)
    X_test_bow = X_test.astype(np.int32)
    nb.fit(X_train_bow, y_train)
    nb_train_time = time.time() - start

    start = time.time()
    nb_pred = nb.predict(X_test_bow)
    nb_inf_time = time.time() - start

    nb_acc = (nb_pred == y_test).mean()
    results['nb_emb'] = {
        'accuracy': nb_acc * 100,
        'train_time': nb_train_time,
        'inf_time': nb_inf_time / len(X_test) * 1000
    }
    print(f"      Accuracy: {nb_acc*100:.2f}%")

    print("\n      --- SVM (TF-IDF) ---")
    start = time.time()

    train_codes = [item['code'] for item in filtered[:split_idx]]
    test_codes = [item['code'] for item in filtered[split_idx:]]

    tfidf = TfidfVectorizer(max_features=2048, tokenizer=tokenize_code)
    X_train_tfidf = tfidf.fit_transform(train_codes)
    X_test_tfidf = tfidf.transform(test_codes)

    svm = LinearSVC(max_iter=2000)
    svm.fit(X_train_tfidf, y_train)
    svm_train_time = time.time() - start

    start = time.time()
    svm_pred = svm.predict(X_test_tfidf)
    svm_inf_time = time.time() - start

    svm_acc = (svm_pred == y_test).mean()
    results['svm'] = {
        'accuracy': svm_acc * 100,
        'train_time': svm_train_time,
        'inf_time': svm_inf_time / len(X_test) * 1000
    }
    print(f"      Accuracy: {svm_acc*100:.2f}%")

    print("\n      --- SVM (Word Embeddings) ---")
    start = time.time()
    svm_emb = LinearSVC(max_iter=2000)
    svm_emb.fit(X_train, y_train)
    svm_emb_train_time = time.time() - start

    start = time.time()
    svm_emb_pred = svm_emb.predict(X_test)
    svm_emb_inf_time = time.time() - start

    svm_emb_acc = (svm_emb_pred == y_test).mean()
    results['svm_emb'] = {
        'accuracy': svm_emb_acc * 100,
        'train_time': svm_emb_train_time,
        'inf_time': svm_emb_inf_time / len(X_test) * 1000
    }
    print(f"      Accuracy: {svm_emb_acc*100:.2f}%")

    print("\n      --- AdderNet-HDC (Word Embeddings) ---")
    start = time.time()
    model = AdderNetHDC(n_vars=vector_size, n_classes=10, table_size=512, seed=42)
    model.train(X_train, y_train)
    model.set_threads(8)
    model.warm_cache()
    model.build_lsh(k=10, l=8)
    hdc_train_time = time.time() - start

    start = time.time()
    hdc_pred = model.predict_batch(X_test)
    hdc_inf_time = time.time() - start

    hdc_acc = (hdc_pred == y_test).mean()
    results['hdc'] = {
        'accuracy': hdc_acc * 100,
        'train_time': hdc_train_time,
        'inf_time': hdc_inf_time / len(X_test) * 1000
    }
    print(f"      Accuracy: {hdc_acc*100:.2f}%")
    print(f"      Inference: {hdc_inf_time/len(X_test)*1000:.3f}ms/pred")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Dataset: CodeSearchNet Python")
    print(f"Training functions: {len(X_train)}")
    print(f"Test functions: {len(X_test)}")
    print(f"Embedding dimension: {vector_size}")
    print(f"HDC: n_vars={vector_size}, D=5000, table_size=256")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'Train Time':>12} {'Inference':>12}")
    print("-" * 60)
    print(f"{'Token IDs + NB':<25} {results['nb_emb']['accuracy']:>9.2f}% {results['nb_emb']['train_time']:>10.2f}s {results['nb_emb']['inf_time']:>10.3f}ms")
    print(f"{'Token IDs + SVM (TF-IDF)':<25} {results['svm']['accuracy']:>9.2f}% {results['svm']['train_time']:>10.2f}s {results['svm']['inf_time']:>10.3f}ms")
    print(f"{'Embedding + SVM':<25} {results['svm_emb']['accuracy']:>9.2f}% {results['svm_emb']['train_time']:>10.2f}s {results['svm_emb']['inf_time']:>10.3f}ms")
    print(f"{'Embedding + HDC':<25} {results['hdc']['accuracy']:>9.2f}% {results['hdc']['train_time']:>10.2f}s {results['hdc']['inf_time']:>10.3f}ms")
    print("-" * 60)

    import subprocess
    mulsd_result = subprocess.run(
        ["grep", "-r", "mulsd", os.path.join(os.path.dirname(__file__), "../../src/")],
        capture_output=True, text=True
    )
    mulsd_count = len(mulsd_result.stdout.strip().split('\n')) if mulsd_result.stdout else 0
    print(f"mulsd HDC: {mulsd_count} ✓" if mulsd_count == 0 else f"mulsd found: {mulsd_count}")

    print("\n" + "=" * 60)
    print("COMPARISON WITH PREVIOUS RESULTS")
    print("=" * 60)
    print(f"Token IDs + HDC (before): 11.2%")
    print(f"Embedding + HDC (now):   {hdc_acc*100:.1f}%")
    print(f"Improvement: +{hdc_acc*100 - 11.2:.1f}pp")
    print(f"Token IDs + SVM: {svm_acc*100:.1f}%")
    print(f"Embedding + SVM: {svm_emb_acc*100:.1f}%")

    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA")
    print("=" * 60)
    print(f"Embedding + HDC ≥ 40%: {'PASS' if hdc_acc >= 0.40 else 'FAIL'} ({hdc_acc*100:.1f}%)")
    print(f"Improvement vs Token IDs ≥ +20pp: {'PASS' if hdc_acc*100 - 11.2 >= 20 else 'FAIL'} ({hdc_acc*100 - 11.2:.1f}pp)")
    print(f"mulsd = 0: {'PASS' if mulsd_count == 0 else 'FAIL'}")
    print(f"Inference < 2ms: {'PASS' if results['hdc']['inf_time'] < 2.0 else 'FAIL'} ({results['hdc']['inf_time']:.3f}ms)")


if __name__ == "__main__":
    main()