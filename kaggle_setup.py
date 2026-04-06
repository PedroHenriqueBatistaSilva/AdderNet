# ╔══════════════════════════════════════════════════════════════════╗
# ║  ADDERGPT-2 — Setup Kaggle v1.4.1 (CUDA 12.8 + T4 fix)          ║
# ║  Compila → Install-libs → sys.path → Treina com Checkpoint      ║
# ╚══════════════════════════════════════════════════════════════════╝

import os, sys, json, time, gc, subprocess, shutil, math
import numpy as np
from pathlib import Path
import psutil

# ───────────────────────────────────────────────────────────────────
# 1. CONFIGURAÇÕES
# ───────────────────────────────────────────────────────────────────
SAMPLES         = 200_000
N_ITER          = 40
HV_DIM          = 2048
CONTEXT_SIZE    = 32
VOCAB_SIZE      = 8000

USE_GPU_INFERENCE = True
USE_GPU_TRAINING  = True
CHECKPOINT_EVERY = 5
KEEP_LAST_N      = 3
RESUME_FROM_CKPT = True

KAGGLE_DIR = Path("/kaggle/working/AdderNet")
MODEL_DIR  = Path("/kaggle/working/addergpt_model")
CKPT_DIR   = MODEL_DIR / "checkpoints"
BUILD_DIR  = KAGGLE_DIR / "build"

MODEL_DIR.mkdir(exist_ok=True, parents=True)
CKPT_DIR.mkdir(exist_ok=True, parents=True)

VERBOSE_LEVEL = 5

# ───────────────────────────────────────────────────────────────────
# 2. SETUP ROBUSTO: Compila + instala libs no pacote Python
# ───────────────────────────────────────────────────────────────────
print("🔧 [Setup] AdderNet v1.4.1 — Compilação Direta (CUDA 12.8 fix)")

if not KAGGLE_DIR.exists():
    print("📦 Clonando AdderNet...")
    # fmt: off
    import subprocess as _sp
    _sp.run(["git", "clone", "--depth", "1",
             "https://github.com/PedroHenriqueBatistaSilva/AdderNet.git",
             str(KAGGLE_DIR)], check=True)
    # fmt: on

# Configura ambiente CUDA
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:/usr/local/nvidia/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

# ── Build CPU + CUDA + copia .so para dentro do pacote addernet/ ──
print("🛠️  Compilando bibliotecas base (CPU)...")
res = subprocess.run(["make", "-C", str(KAGGLE_DIR), "all"],
                     capture_output=True, text=True)
if res.returncode != 0:
    print(f"❌ Falha no make all:\n{res.stderr[-500:]}")
    raise RuntimeError("Compilação CPU falhou")
print("✅ Bibliotecas CPU compiladas")

# CUDA 2026
cuda_so = BUILD_DIR / "libaddernet_cuda_2026.so"
if not cuda_so.exists():
    print("🚀 Compilando CUDA 2026 (T4 sm_75)...")
    cuda_env = {**os.environ}
    res = subprocess.run(
        ["make", "-C", str(KAGGLE_DIR), "cuda_2026"],
        capture_output=True, text=True, timeout=300,
        env=cuda_env,
    )
    if res.returncode == 0 and cuda_so.exists():
        print("✅ CUDA 2026 compilado")
    else:
        print(f"⚠️  CUDA indisponível (rc={res.returncode}). Fallback CPU.")
        if res.stderr:
            print(f"   stderr: {res.stderr[-500:]}")
        USE_GPU_INFERENCE = USE_GPU_TRAINING = False

# 🔑 Copia .so de build/ para addernet/ (o __init__.py também faz auto-copy)
print("🔗 Instalando libs no pacote Python...")
res2 = subprocess.run(
    ["make", "-C", str(KAGGLE_DIR), "install-libs"],
    capture_output=True, text=True
)
if res2.returncode != 0:
    print(f"   ⚠️  install-libs falhou, tentando copy manual...")
    for lib in BUILD_DIR.glob("*.so"):
        dest = KAGGLE_DIR / "addernet" / lib.name
        shutil.copy2(lib, dest)
        print(f"   📄 {lib.name} → addernet/")
else:
    print(f"   {res2.stdout.strip()}")

# Instala deps Python
print("📦 Instalando dependências Python...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "tokenizers", "datasets", "psutil"],
    capture_output=True, text=True
)

# Adiciona ao path e importa
sys.path.insert(0, str(KAGGLE_DIR))
try:
    from addernet import AdderNetHDC, AdderAttention
    try:
        from addernet import hdc_detect_backend
    except ImportError:
        def hdc_detect_backend(): return "CPU"
    print("✅ AdderNet importado com sucesso!")
except OSError as e:
    print(f"❌ Falha ao carregar bibliotecas: {e}")
    print("💡 Listando addernet/:")
    for f in (KAGGLE_DIR / "addernet").glob("*.so"):
        print(f"   {f.name} ({f.stat().st_size} bytes)")
    raise

def now(): return time.strftime("%H:%M:%S")
def ram_gb(): return psutil.virtual_memory().used / (1024**3)

print("\n" + "="*70)
print(f"  [{now()}] 🧠 AdderGPT-2 v1.4.1 | Kaggle + 2x T4")
print(f"  Backend: {hdc_detect_backend()} | HV_DIM: {HV_DIM}")
print(f"  GPU Train: {USE_GPU_TRAINING} | GPU Infer: {USE_GPU_INFERENCE}")
print("="*70 + "\n")

# ───────────────────────────────────────────────────────────────────
# 3. TOKENIZER & DATASET
# ───────────────────────────────────────────────────────────────────
print(f"[{now()}] [1/5] Carregando Wikitext-103...")
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

tok_path = MODEL_DIR / "tokenizer.json"
if not tok_path.exists():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
    def batch_iterator(bs=1000):
        for i in range(0, len(dataset), bs):
            yield dataset[i:i+bs]["text"]
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save(str(tok_path))
else:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))

BOS, EOS, PAD = [tokenizer.token_to_id(t) for t in ["[BOS]", "[EOS]", "[PAD]"]]

# ───────────────────────────────────────────────────────────────────
# 4. EMBEDDINGS & FEATURE EXTRACTION
# ───────────────────────────────────────────────────────────────────
print(f"[{now()}] [2/5] Gerando embeddings HDC ({VOCAB_SIZE} × {HV_DIM})...")
emb_path = MODEL_DIR / "vocab_embeddings.npy"
if not emb_path.exists():
    rng = np.random.default_rng(42)
    vocab_emb = rng.choice([-1, 1], size=(VOCAB_SIZE, HV_DIM)).astype(np.float32)
    np.save(emb_path, vocab_emb)
else:
    vocab_emb = np.load(emb_path)

print(f"[{now()}] [3/5] Extraindo {SAMPLES} janelas com AdderAttention...")
attention = AdderAttention(threshold=None)
X_train = np.zeros((SAMPLES, HV_DIM), dtype=np.float64)
y_train = np.zeros((SAMPLES,), dtype=np.int32)

sample_idx = 0
for row in dataset:
    if sample_idx >= SAMPLES: break
    text = row["text"].strip()
    if not text: continue
    ids = tokenizer.encode(text).ids
    if len(ids) < 2: continue
    for i in range(1, len(ids)):
        if sample_idx >= SAMPLES: break
        window = ids[max(0, i - CONTEXT_SIZE):i]
        if len(window) < CONTEXT_SIZE:
            window = [BOS] * (CONTEXT_SIZE - len(window)) + window
        ctx_vecs = vocab_emb[window]
        Q = ctx_vecs[-1:].reshape(1, 1, HV_DIM)
        K = ctx_vecs.reshape(1, CONTEXT_SIZE, HV_DIM)
        V = K.copy()
        context_vec = attention.forward(Q, K, V).squeeze()
        x_norm = ((context_vec / CONTEXT_SIZE) + 1.0) / 2.0 * 255.0
        X_train[sample_idx] = np.clip(np.nan_to_num(x_norm), 0, 255).astype(np.float64)
        y_train[sample_idx] = ids[i]
        sample_idx += 1

del dataset, tokenizer, vocab_emb, attention
gc.collect()
print(f"  → RAM: {ram_gb():.1f} GB | Amostras: {sample_idx}")

# ───────────────────────────────────────────────────────────────────
# 5. CHECKPOINT & TREINO
# ───────────────────────────────────────────────────────────────────
def save_checkpoint(model, epoch, val_acc, path):
    model.save(str(path / "model.bin"))
    with open(path / "meta.json", "w") as f:
        json.dump({"epoch": epoch, "val_accuracy": float(val_acc), "timestamp": now()}, f, indent=2)
    print(f"  💾 Checkpoint: epoch={epoch}, acc={val_acc:.2%}")

def load_checkpoint(path):
    if not (path / "model.bin").exists(): return None, 0
    try:
        with open(path / "meta.json") as f: meta = json.load(f)
        model = AdderNetHDC.load(str(path / "model.bin"))
        return model, meta["epoch"]
    except Exception as e:
        print(f"  ⚠️  Erro ao carregar: {e}")
        return None, 0

def cleanup_old(ckpt_dir, keep_n):
    ckpts = sorted(ckpt_dir.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
    while len(ckpts) > keep_n: shutil.rmtree(ckpts.pop(0))

print(f"\n[{now()}] [4/5] Iniciando treino...")
X_train = np.ascontiguousarray(X_train, dtype=np.float64)
y_train = np.ascontiguousarray(y_train, dtype=np.int32)

start_epoch = 0
if RESUME_FROM_CKPT:
    last = sorted(CKPT_DIR.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
    if last: model, start_epoch = load_checkpoint(last[-1])

if start_epoch == 0:
    print(f"  🆕 Criando AdderNetHDC...")
    model = AdderNetHDC(n_vars=HV_DIM, n_classes=VOCAB_SIZE, table_size=256, hv_dim=HV_DIM,
                        use_gpu=USE_GPU_INFERENCE, use_gpu_training=USE_GPU_TRAINING, seed=42)
    model.set_threads(4)

start_time = time.time()
best_acc, no_improve, patience = 0.0, 0, 15

try:
    for epoch in range(start_epoch, N_ITER):
        history = model.train(X=X_train, y=y_train, n_iter=1, lr=1.0, margin='5%',
                              regenerate=0.02, patience=0, interactions=10,
                              verbose=(VERBOSE_LEVEL if (epoch+1)%VERBOSE_LEVEL==0 else 0))
        curr_epoch, curr_acc = epoch + 1, history.get('best_val_accuracy', 0.0)
        if curr_epoch % VERBOSE_LEVEL == 0 or curr_epoch == 1:
            print(f"  📊 Epoch {curr_epoch:2d}/{N_ITER} | Acc: {curr_acc:5.2%} | RAM: {ram_gb():.1f}GB")
        if curr_epoch % CHECKPOINT_EVERY == 0:
            ckpt = CKPT_DIR / f"epoch_{curr_epoch:03d}"
            ckpt.mkdir(exist_ok=True)
            save_checkpoint(model, curr_epoch, curr_acc, ckpt)
            cleanup_old(CKPT_DIR, KEEP_LAST_N)
            model.save(str(MODEL_DIR / "adder_lm_model.bin"))
        if curr_acc > best_acc:
            best_acc, no_improve = curr_acc, 0
            model.save(str(MODEL_DIR / "best_model.bin"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  ⏹️  Early stopping ({patience} épocas)")
                break
except KeyboardInterrupt:
    print(f"\n  ⚠️  Interrompido na época {curr_epoch}")
except Exception as e:
    print(f"\n  ❌  Erro: {e}")
    import traceback; traceback.print_exc()
    model.save(str(MODEL_DIR / "adder_lm_model.bin"))

end_time = time.time()
total_epochs = epoch + 1 if 'epoch' in locals() else start_epoch

# ───────────────────────────────────────────────────────────────────
# 6. SALVAMENTO FINAL
# ───────────────────────────────────────────────────────────────────
print(f"\n[{now()}] [5/5] Salvando...")
model.save(str(MODEL_DIR / "adder_lm_model.bin"))
cfg = {"version": "AdderGPT-2 v1.4.1-Kaggle", "epochs": total_epochs, "best_acc": float(best_acc),
       "gpu": {"train": USE_GPU_TRAINING, "infer": USE_GPU_INFERENCE}}
with open(MODEL_DIR / "config.json", "w") as f: json.dump(cfg, f, indent=2)

zip_path = MODEL_DIR.parent / "addergpt_model.zip"
shutil.make_archive(str(zip_path.with_suffix('')), 'zip', MODEL_DIR)
print(f"\n🏁 Concluído em {(end_time-start_time)/3600:.2f}h | Acc: {best_acc:.2%}")
print(f"📦 Download: {zip_path}")
