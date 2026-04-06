# QWEN.md — AdderNet

## Visão Geral do Projeto

**AdderNet** é uma biblioteca de machine learning que **não usa multiplicação de ponto flutuante** durante a inferência. Em vez disso, utiliza **lookups em tabela (LUT)** e operações de soma inteiras, tornando a inferência viável em microcontroladores sem FPU (ESP32, STM32, RPi).

Versão atual: **1.4.0**

### Componentes Principais

| Classe | Descrição |
|---|---|
| `AdderNetLayer` | Rede de uma variável — LUT + soma, zero multiplicação |
| `AdderNetHDC` | Classificador multivariável — Hyperdimensional Computing (HDC) |
| `AdderCluster` | Ensemble de `AdderNetLayer` com estratégias de combinação |
| `AdderBoost` | Gradient Boosting com `AdderNetLayer` |
| `AdderAttention` | Attention mechanism baseado em distância de Hamming |

### Tecnologias e Dependências

- **Python**: 3.8+
- **Dependências runtime**: `numpy`, `scikit-learn`, `scipy`
- **Dependências dev**: `pytest`
- **Build C**: `gcc`, `make`
- **CUDA (opcional)**: `nvcc` (requer toolkit CUDA 2026)
- **SIMD**: AVX2 (x86_64), NEON (ARM), fallback escalar

---

## Estrutura do Projeto

```
AdderNet/
├── addernet/                  # Pacote Python principal
│   ├── __init__.py            # Auto-build das libs C, detecção CUDA
│   ├── addernet.py            # Wrapper ctypes do AdderNetLayer
│   ├── addernet_hdc.py        # Wrapper ctypes do AdderNetHDC
│   ├── cluster.py             # AdderCluster ensemble
│   ├── boost.py               # AdderBoost gradient boosting
│   ├── attention.py           # AdderAttention mechanism
│   ├── build_ext.py           # Compilação runtime C/CUDA (legado)
│   ├── build_ext_2026.py      # Compilação runtime C/CUDA (novo)
│   ├── cuda_detector.py       # Detecção de GPU CUDA
│   └── src/                   # Código-fonte C/CUDA incluído no pacote
│       ├── addernet.c/h       # LUT single-variable
│       ├── addernet_hdc.c/h   # HDC multivariável
│       ├── hdc_core.c/h       # Hipervetores (XOR, bundling, Hamming)
│       ├── hdc_lsh.c/h        # Locality-Sensitive Hashing
│       ├── hdc_core_cuda.c    # Kernels CUDA (inline PTX)
│       └── hdc_cuda_batch.c   # Batch prediction CUDA
│
├── src/cuda_train/            # Kernels CUDA de treinamento (2026)
│   └── addernet_hdc_train_cuda_2026.cu
│
├── tests/                     # Testes unitários C
│   ├── test_main.c
│   └── test_hdc_main.c
│
├── test_validation.py         # Testes de validação Python
├── test_attention.py          # Testes do attention mechanism
├── Makefile                   # Build C/CUDA
├── setup.py                   # Setuptools com MakeBuildExt custom
├── pyproject.toml             # Metadados do pacote
└── docs/                      # Documentação
```

---

## Build e Execução

### Compilar bibliotecas C

```bash
make all              # Compila libaddernet.so e libaddernet_hdc.so
make addernet         # Apenas libaddernet.so
make hdc              # Apenas libaddernet_hdc.so
```

### Compilar backends CUDA (opcionais)

```bash
make cuda             # Inline PTX via gcc (sem nvcc)
make cuda_native      # Requer nvcc → libaddernet_cuda.so
make cuda_2026        # Kernel cooperativo Ampere+ → libaddernet_cuda_2026.so
```

### Instalar pacote Python

```bash
# Modo desenvolvimento (editable)
pip install -e .

# Build e instalação normal
pip install .
```

### Testes

```bash
make test             # Compila e roda testes C
make test_addernet    # Testes C do AdderNetLayer
make test_hdc         # Testes C do AdderNetHDC
python test_validation.py   # Validação Python
python test_attention.py    # Testes attention
pytest -v             # Suite pytest
```

### Limpar build artifacts

```bash
make clean
```

---

## Convenções de Desenvolvimento

### Arquitetura C

- **AdderNetLayer**: Inferência via `offset_table[(int_input + bias) & mask]` — uma leitura de memória, zero aritmética
- **Treinamento**: Busca direcional por tentativa e erro (sem gradientes/backprop). Ajusta entradas da tabela por `+/- lr`
- **AdderNetHDC**: Encoding via hipervetores determinísticos gerados a partir de seeds; treinamento OnlineHD (single-pass) e AdaptHD (iterativo)
- **Tamanho da tabela**: Deve ser potência de 2 (default 256) para masking eficiente

### Backend CUDA 2026

- **Kernel cooperativo Ampere+**: Shared memory 100KB, warp-level primitives, unified kernel
- **Kernel selection automático**: Detecta GPU e seleciona kernel otimizado (Ampere sm_80+ → Turing sm_70-75 → Legacy sm_61)
- **Unified Memory**: Zero-copy para datasets pequenos (`ADDERNET_UNIFIED_MEMORY=1`)
- **CUDA Graphs**: Capture once, replay many (`ADDERNET_CUDA_GRAPHS=1`)
- **Persistent Kernel**: Elimina overhead de kernel launch (`ADDERNET_PERSISTENT_KERNEL=1`)

### Python

- Wrappers via `ctypes` para as bibliotecas C compartilhadas
- Auto-build no `__init__.py` se `.so` não encontrado (útil para Colab/runtime)
- Prioriza `build_ext_2026`, fallback para `build_ext` legado

### Platform-Specific

| Plataforma | Extensão | Flags SIMD |
|---|---|---|
| Linux x86_64 | `.so` | `-mavx2 -mpopcnt -march=native` |
| Linux ARM64 | `.so` | `-march=armv8-a+simd -mfpu=neon` |
| macOS | `.dylib` | scalar |
| Windows | `.dll` | scalar |

---

## Limitações Conhecidas

- **AdderNetLayer**: apenas uma variável de entrada por camada
- **AdderNetHDC**: acurácia inferior a MLPs profundas em datasets complexos (trade-off por zero multiplicação)
- `hv_dim` muito pequeno (< 1000) pode colapsar a acurácia
- Valores de entrada são truncados para `int` internamente (partes fracionárias perdidas)
- Formato binário de save não tem versionamento

---

## Licença

Apache 2.0 — © Pedro Henrique Batista Silva
