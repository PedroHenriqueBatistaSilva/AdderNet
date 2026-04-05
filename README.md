# AdderNet

[![PyPI version](https://img.shields.io/pypi/v/addernet.svg)](https://pypi.org/project/addernet/)
[![Python](https://img.shields.io/pypi/pyversions/addernet.svg)](https://pypi.org/project/addernet/)
[![License](https://img.shields.io/github/license/PedroHenriqueBatistaSilva/AdderNet.svg)](LICENSE)

Biblioteca de machine learning que **não usa multiplicação de ponto flutuante** na inferência. Zero.

> Benchmarks medidos em CPU x86-64 com backend **AVX2** e GPUs NVIDIA via **CUDA**, Python 3.x, v1.2.5.

---

## O que é?

AdderNet substitui multiplicações por **lookups em tabela** (LUT) e operações de soma inteiras,
tornando a inferência viável em microcontroladores sem FPU (ESP32, STM32, RPi).

A biblioteca expõe quatro componentes principais:

| Classe | Descrição |
|---|---|
| `AdderNetLayer` | Rede de uma variável — LUT + soma, zero multiplicação |
| `AdderNetHDC` | Classificador multivariável — Hyperdimensional Computing (HDC) |
| `AdderCluster` | Ensemble de `AdderNetLayer` com estratégias de combinação |
| `AdderBoost` | Gradient Boosting com `AdderNetLayer` — inferência sem multiplicação |

---

## Novidades v1.2.5 🚀

- **HV_DIM Dinâmico**: A dimensionalidade hiperdimensional (`hv_dim`) agora é configurável em tempo de execução (`512`, `1024`, `2048`, `4096`, etc), sem precisar recompilar a biblioteca C!
- **Aceleração CUDA NATIVA no Treinamento**: O loop iterativo de retreino (`AdaptHD`) foi reescrito em CUDA para ser executado massivamente em paralelo usando `atomicAdd` e otimizações de bitwise. Basta usar `use_gpu_training=True`.
- **Aceleração CUDA na Inferência**: O processo de `predict_batch` agora tem suporte a GPU via kernels CUDA dedicados. Basta instanciar o modelo com `use_gpu=True`.
- **Compatibilidade e Fallback**: O pacote compila nativamente o C++ e CUDA no momento do `pip install`. Se a máquina alvo (como um Raspberry Pi) não tiver placa de vídeo NVIDIA, a biblioteca roda perfeitamente fazendo fallback silencioso para CPU com `AVX2`, `NEON` ou `SCALAR`.
- Correção de bugs de Ctypes FFI e memory alignment (agora todos os `aligned_alloc` usam arrays flats dinâmicos, preservando compatibilidade absoluta entre os tensores C contíguos e o numpy).

---

## Instalação

```bash
pip install addernet
```

Ou do código-fonte (para compilar com otimizações nativas e CUDA opcional):

```bash
git clone https://github.com/PedroHenriqueBatistaSilva/AdderNet.git
cd AdderNet
make all         # Compila binários da CPU
make cuda_native # Opcional: Compila o backend de GPU (requer nvcc)
pip install -e .
```

---

## Uso — AdderNetLayer (uma variável)

```python
from addernet import AdderNetLayer

rede = AdderNetLayer(size=256, bias=50, input_min=-50, input_max=200, lr=0.1)

celsius    = [0, 10, 20, 25, 30, 37, 50, 80, 100]
fahrenheit = [32, 50, 68, 77, 86, 98.6, 122, 176, 212]

rede.train(celsius, fahrenheit)

print(rede.predict(37))    # 98.60
print(rede.predict(100))   # 212.00
```

### Previsão em lote (numpy)

```python
import numpy as np

entradas = np.linspace(-50, 200, 1_000_000, dtype=np.float64)
saidas = rede.predict_batch(entradas)   # ~178M pred/s com AVX2
```

---

## Uso — AdderNetHDC (Aceleração GPU e HDC Dinâmico)

```python
from addernet import AdderNetHDC
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
X = MinMaxScaler(feature_range=(0, 150)).fit_transform(iris.data)
y = iris.target

# HV_DIM dinâmico configurável (ex: 2048, 4096)
model = AdderNetHDC(
    n_vars=4, 
    n_classes=3, 
    table_size=256, 
    hv_dim=4096,              # <- Dimensionalidade configurável no runtime!
    use_gpu=True,             # <- Ativa inferência batch em CUDA
    use_gpu_training=True     # <- Ativa treinamento iterativo em CUDA
)

# Arrays numpy precisam ser "C Contiguous" 
X_c = np.ascontiguousarray(X, dtype=np.float64)
y_c = np.ascontiguousarray(y, dtype=np.int32)

# Treino single-pass (OnlineHD)
model.train(X_c, y_c)

# Retreino iterativo (AdaptHD) — massivamente paralelo na GPU
model.train(X_c, y_c, n_iter=20, lr=1.0)

# Inferência massiva e ultrarrápida via GPU
preds = model.predict_batch(X_c)

print(f"Acurácia: {model.accuracy(X_c, y_c)*100:.1f}%")
```

---

## Uso — AdderCluster (ensemble multi-nó)

```python
from addernet import AdderCluster
import numpy as np

cluster = AdderCluster(
    n_nodes=4,
    strategy='feature',    # 'random' | 'range' | 'feature' | 'boosting'
    combination='vote',    # 'vote' | 'mean' | 'stack'
    input_min=0,
    input_max=150,
)

cluster.fit(X, y)
preds = cluster.predict_batch(X)

cluster.info()
```

---

## Otimizações disponíveis

```python
from addernet import hdc_detect_backend

print(hdc_detect_backend())   # 'AVX2', 'NEON', ou 'SCALAR'

model.set_threads(4)      # multithreading CPU (AdderNetHDC)
model.warm_cache()        # pré-computar hipervectors
model.set_cache(False)    # desligar cache (hardware com pouca RAM)
```

---

## Limitações

- **AdderNetLayer**: apenas uma variável de entrada por camada
- **AdderNetHDC**: acurácia inferior a MLPs profundas em datasets complexos (troca por zero multiplicação)
- `hv_dim` muito pequeno (< 1000) pode colapsar a acurácia, use a Dimensionalidade Dinâmica para testar!

---

## Licença

[Apache 2.0](LICENSE) — © Pedro Henrique Batista Silva