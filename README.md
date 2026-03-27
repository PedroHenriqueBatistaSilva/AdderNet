# AdderNet

Biblioteca de machine learning que **não usa multiplicação** na inferência. Zero.

## O que é?

AdderNet tem duas camadas:

- **AdderNet** — funções de uma variável, lookup table, zero multiplicação
- **AdderNet-HDC** — múltiplas variáveis, Hyperdimensional Computing,
  generalização por bundle, criatividade com temperatura

Novidades na v1.0.5:
- **OnlineHD** — treino com bundling ponderado por novidade (anti-saturação)
- **AdaptHD** — retreino iterativo com correção de erro (`n_iter`)
- **D=2500** — inferência 4x mais rápida vs D=10000 anterior
- **Early-exit Hamming** — aborta comparação cedo quando classes são separadas
- Acurácia: Breast Cancer ~82% → 93%, Wine mantém 90%+

## Instalação

```bash
pip install addernet
```

Ou do código-fonte:

```bash
git clone ...
cd addernet_lib
make
pip install -e .
```

## Uso — Uma variável

```python
from addernet import AdderNetLayer

# Criar a rede
rede = AdderNetLayer(size=256, bias=50, input_min=-50, input_max=200, lr=0.1)

# Dados de treino: Celsius -> Fahrenheit
celsius    = [0, 10, 20, 25, 30, 37, 50, 80, 100]
fahrenheit = [32, 50, 68, 77, 86, 98.6, 122, 176, 212]

# Treinar
rede.train(celsius, fahrenheit)

# Prever
print(rede.predict(37))   # 98.60
print(rede.predict(100))  # 212.00
```

### Salvar e carregar

```python
rede.save("meu_modelo.bin")
rede = AdderNetLayer.load("meu_modelo.bin")
print(rede.predict(37))  # 98.60
```

### Previsão em lote (numpy)

```python
import numpy as np

entradas = np.array([0, 25, 37, 100], dtype=np.float64)
saidas = rede.predict_batch(entradas)
# array([32.0, 77.0, 98.6, 212.0])
```

## Uso — AdderNet-HDC (múltiplas variáveis)

```python
from addernet import AdderNetHDC
import numpy as np
from sklearn.datasets import load_iris

# Carregar dados
iris = load_iris()
X, y = iris.data, iris.target

# Criar modelo (4 variáveis, 3 classes)
model = AdderNetHDC(n_vars=4, n_classes=3, table_size=256, seed=42)

# Treinar (single-pass com OnlineHD anti-saturação)
model.train(X, y)

# Treinar com retreino iterativo (AdaptHD) — melhora acurácia
model.train(X, y, n_iter=20, lr=1.0)

# Prever
pred = model.predict(X[0])
print(f"Classe predita: {pred}")

# Batch com múltiplas threads
preds = model.predict_batch(X)

# Ativar cache para máxima velocidade
model.warm_cache()
preds = model.predict_batch(X)  # ~3x mais rápido

# Salvar e carregar
model.save("iris_model.bin")
model = AdderNetHDC.load("iris_model.bin")
```

### Retreino iterativo (AdaptHD)

O parâmetro `n_iter` ativa correção de erro iterativa após o treino inicial.
A cada iteração, o modelo prediz no conjunto de treino e corrige os erros:
reforça a classe certa e penaliza a classe errada no espaço de hipervectores.

```python
# n_iter=0  → single-pass (OnlineHD bundling, padrão)
# n_iter=20 → 20 passadas de correção de erro
# lr controla a intensidade da correção (default=1.0)
model.train(X, y, n_iter=20, lr=1.0)
```

Recomendações:
- `n_iter=20, lr=1.0` — bom default para maioria dos datasets
- `n_iter=50, lr=0.5` — mais conservador, para datasets pequenos
- `n_iter=0` — só OnlineHD, sem correção iterativa

## Generalização e criatividade

```python
# Misturar dois conceitos — gera algo novo sem ver dados
roupa_nova = model.bundle_classes([0, 4])  # camiseta + casaco

# Temperatura — variações controladas
for temp in [0.0, 0.1, 0.2, 0.3, 0.5]:
    variacao = model.add_noise(model.codebook[0], temp)
    print(f"temp={temp} → {model.classify_hv(variacao)}")
```

## Otimizações disponíveis

```python
from addernet import AdderNetHDC, hdc_detect_backend
print(hdc_detect_backend())  # 'AVX2', 'NEON', ou 'SCALAR'

model.set_threads(8)      # multithreading
model.warm_cache()        # pré-computar hipervectors
model.set_cache(False)    # desligar cache (hardware com pouca RAM)
```

## Performance (v1.0.5, D=2500)

```
AdderNet (uma variável):
  C lote numpy:           ~247M pred/s

AdderNet-HDC (Wine, 13 vars):
  Baseline (1 thread):    ~630 pred/s
  Cache + 4 threads:      ~1.600 pred/s
  Acurácia:               90.7%
  mulsd em inferência:    0

AdderNet-HDC (Breast Cancer, 30 vars):
  Baseline (1 thread):    ~112 pred/s
  Cache + 4 threads:      ~344 pred/s
  Acurácia:               93.0% (n_iter=20)
  mulsd em inferência:    0
```

Para melhor performance em Python, use `predict_batch()` com arrays numpy.

## Casos de uso

- Sensores industriais com hardware limitado (sem FPU)
- Classificação em tempo real embarcada (ESP32, STM32, RPi)
- Privacidade: dados originais não precisam ser guardados pós-treino
- One-shot learning: aprende classe nova com uma única amostra

## Exemplos prontos

```bash
cd addernet_lib

# Celsius -> Fahrenheit (uma variável)
python3 examples/basic/celsius_fahrenheit.py

# AdderNet-HDC com Iris
python3 examples/hdc/iris_hdc.py

# Benchmark HDC
python3 examples/hdc/benchmark_hdc.py
```

## Estrutura de arquivos

```
addernet_lib/
├── src/              ← código C otimizado (AVX2, NEON)
├── python/           ← bindings Python
├── addernet/         ← pacote instalável (src + .so + bindings)
├── tests/
├── examples/
│   ├── basic/        ← AdderNet básica
│   └── hdc/          ← AdderNet-HDC
├── build/
└── Makefile          ← detecção automática de plataforma
```

## Limitações

- **AdderNet básica**: só uma variável de entrada
- **AdderNet-HDC**: acurácia inferior a MLP em datasets complexos
  (~90-93% vs ~98%), mas sem multiplicação de ponto flutuante
- Contexto sequencial ainda não implementado (LLM embarcado em desenvolvimento)
- D muito pequeno (< 1000) colapsa a acurácia
