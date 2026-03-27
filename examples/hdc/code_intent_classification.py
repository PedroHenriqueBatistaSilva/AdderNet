#!/usr/bin/env python3
"""
Code Intent Classification with AdderNet-HDC
===========================================
Classifies Python function intent using AdderNet-HDC and compares with baselines.
"""

import os
import sys
import time
import tokenize
import io
from collections import Counter

import numpy as np
from datasets import load_dataset
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
    """Classify intent based on keywords in function name and docstring."""
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


def build_vocabulary(dataset, table_size=2048):
    """Build vocabulary from tokenized code."""
    all_tokens = []

    for item in dataset:
        if isinstance(item, dict):
            code = item.get('func_code_string', '')
        else:
            code = item
        if code:
            tokens = tokenize_code(code)
            all_tokens.extend(tokens)

    token_counts = Counter(all_tokens)
    most_common = token_counts.most_common(table_size - 1)

    token2id = {'<PAD>': 0}
    for i, (token, _) in enumerate(most_common, start=1):
        token2id[token] = i

    return token2id


def encode_function(code, token2id, n_vars=50):
    """Encode function using token IDs."""
    tokens = tokenize_code(code)
    
    token_ids = [token2id.get(t, 0) for t in tokens[:n_vars]]
    
    while len(token_ids) < n_vars:
        token_ids.append(0)
    
    return [float(x) for x in token_ids[:n_vars]]


def main():
    print("=" * 60)
    print("Code Intent Classification — AdderNet-HDC")
    print("=" * 60)

    np.random.seed(42)

    print("\n[1/6] Loading CodeSearchNet dataset...")
    ds = load_dataset("code_search_net", "python", split="train")
    print(f"      Total functions: {len(ds)}")

    print("\n[2/6] Filtering and labeling functions...")
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

    print("\n[3/6] Building vocabulary...")
    sample_codes = [item['code'] for item in filtered]
    token2id = build_vocabulary(sample_codes, table_size=2048)
    print(f"      Vocabulary size: {len(token2id)}")

    print("\n[4/6] Encoding functions...")
    X = np.array([encode_function(item['code'], token2id, 50) for item in filtered], dtype=np.float64)
    y = np.array([item['intent'] for item in filtered], dtype=np.int32)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"      Training: {len(X_train)}, Test: {len(X_test)}")
    print(f"      Class distribution: {np.bincount(y_train)}")

    results = {}

    print("\n[5/6] Training and evaluating models...")

    print("\n      --- Naive Bayes ---")
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
    results['nb'] = {
        'accuracy': nb_acc * 100,
        'train_time': nb_train_time,
        'inf_time': nb_inf_time / len(X_test) * 1000
    }
    print(f"      Accuracy: {nb_acc*100:.2f}%")
    print(f"      Train time: {nb_train_time:.2f}s")
    print(f"      Inference: {nb_inf_time/len(X_test)*1000:.3f}ms/pred")

    print("\n      --- SVM (TF-IDF) ---")
    start = time.time()

    train_codes = [item['code'] for item in filtered[:split_idx]]
    test_codes = [item['code'] for item in filtered[split_idx:]]

    tfidf = TfidfVectorizer(max_features=2048, tokenizer=tokenize_code)
    X_train_tfidf_full = tfidf.fit_transform(train_codes)
    X_test_tfidf = tfidf.transform(test_codes)

    svm = LinearSVC(max_iter=2000)
    svm.fit(X_train_tfidf_full, y_train)
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
    print(f"      Train time: {svm_train_time:.2f}s")
    print(f"      Inference: {svm_inf_time/len(X_test)*1000:.3f}ms/pred")

    print("\n      --- AdderNet-HDC ---")
    start = time.time()
    model = AdderNetHDC(n_vars=50, n_classes=10, table_size=2048, seed=42)
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
    print(f"      Train time: {hdc_train_time:.2f}s")
    print(f"      Inference: {hdc_inf_time/len(X_test)*1000:.3f}ms/pred")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Dataset: CodeSearchNet Python")
    print(f"Training functions: {len(X_train)}")
    print(f"Test functions: {len(X_test)}")
    print(f"Classes: 10 intents")
    print(f"Tokens/function: 50")
    print(f"D=5000, K=10, L=8")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Train Time':>12} {'Inference':>12}")
    print("-" * 60)
    print(f"{'Naive Bayes':<20} {results['nb']['accuracy']:>9.2f}% {results['nb']['train_time']:>10.2f}s {results['nb']['inf_time']:>10.3f}ms")
    print(f"{'SVM (TF-IDF)':<20} {results['svm']['accuracy']:>9.2f}% {results['svm']['train_time']:>10.2f}s {results['svm']['inf_time']:>10.3f}ms")
    print(f"{'AdderNet-HDC':<20} {results['hdc']['accuracy']:>9.2f}% {results['hdc']['train_time']:>10.2f}s {results['hdc']['inf_time']:>10.3f}ms")
    print("-" * 60)

    import subprocess
    mulsd_result = subprocess.run(
        ["grep", "-r", "mulsd", os.path.join(os.path.dirname(__file__), "../../src/")],
        capture_output=True, text=True
    )
    mulsd_count = len(mulsd_result.stdout.strip().split('\n')) if mulsd_result.stdout else 0
    print(f"mulsd HDC: {mulsd_count} ✓" if mulsd_count == 0 else f"mulsd found: {mulsd_count}")

    print("\n" + "=" * 60)
    print("DEMONSTRATION")
    print("=" * 60)

    test_snippets = [
        ("def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr", "sort_search"),
        ("def calculate_mean(values):\n    return sum(values) / len(values)", "math"),
        ("def read_json_file(path):\n    with open(path) as f:\n        return json.load(f)", "io"),
        ("def validate_email(email):\n    return '@' in email and '.' in email", "validation"),
        ("def encode_base64(data):\n    return base64.b64encode(data)", "conversion"),
    ]

    correct = 0
    for snippet, expected in test_snippets:
        encoded = encode_function(snippet, token2id, 50)
        pred = model.predict(encoded)
        pred_name = INTENT_CLASSES[pred]
        status = "✓" if pred_name == expected else "✗"
        if pred_name == expected:
            correct += 1
        print(f"{snippet:40s} → {pred_name:<12} {status}")

    print(f"\nDemonstration: {correct}/5 correct")

    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA")
    print("=" * 60)
    print(f"AdderNet-HDC accuracy ≥ 70%: {'PASS' if hdc_acc >= 0.70 else 'FAIL'} ({hdc_acc*100:.1f}%)")
    print(f"vs Naive Bayes ≥ equal: {'PASS' if hdc_acc >= nb_acc else 'FAIL'}")
    print(f"Inference < 1ms: {'PASS' if results['hdc']['inf_time'] < 1.0 else 'FAIL'} ({results['hdc']['inf_time']:.3f}ms)")
    print(f"mulsd = 0: {'PASS' if mulsd_count == 0 else 'FAIL'}")
    print(f"Demonstration ≥ 4/5: {'PASS' if correct >= 4 else 'FAIL'} ({correct}/5)")


if __name__ == "__main__":
    main()
