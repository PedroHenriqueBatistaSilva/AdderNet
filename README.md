# AdderNet Learning Edition — 1.5.0.post1

Uma edição corrigida e pedagógica da AdderNet: aprendizagem por tabelas de consulta (LUT), Hyperdimensional Computing (HDC), ensembles aditivos e attention por distância L1.

> **Honestidade técnica:** a inferência de uma `AdderNetLayer` treinada é essencialmente indexação de tabela. Quantização, HDC, attention, pré-processamento e componentes Python podem usar outras operações. “Zero multiplicação” deve ser entendido como propriedade do caminho LUT central, não de todo pipeline possível.

## Instalação

```bash
python -m pip install .
addernet-selftest
```

Ao importar diretamente do código-fonte, a biblioteca tenta compilar as extensões CPU com `gcc`, `clang` ou `cc`. Desative isso com:

```bash
ADDERNET_AUTOBUILD=0 python seu_script.py
```

Para otimizações nativas da CPU local:

```bash
ADDERNET_NATIVE=1 addernet-build
```

## Primeira LUT

```python
from addernet import AdderNetLayer

layer = AdderNetLayer(size=256, bias=0, input_min=0, input_max=100, lr=0.1)
layer.train([0, 10, 20, 30], [32, 50, 68, 86])
print(layer.predict(25))
```

A camada converte a entrada para inteiro, calcula:

```text
index = (int(input) + bias) & (size - 1)
```

E retorna `offset_table[index]`.

## Implementação transparente em Python

```python
from addernet import ReferenceAdderNetLayer

layer = ReferenceAdderNetLayer(size=64, input_min=0, input_max=31, lr=0.25)
layer.train([0, 8, 16, 24, 31], [2, 14, 26, 38, 48.5])
print(layer.offset_table)
```

`ReferenceAdderNetLayer` é deliberadamente lenta, mas permite estudar e depurar o algoritmo sem `ctypes` nem C.

## Regressão multivariada aditiva

```python
import numpy as np
from addernet import AdderNetAdditiveRegressor

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(500, 3))
y = 1.8 * X[:, 0] - 0.5 * X[:, 1] + np.sin(X[:, 2])

model = AdderNetAdditiveRegressor(
    table_size=128,
    backfit_rounds=3,
    epochs_raw=50,
    epochs_expanded=100,
).fit(X[:400], y[:400])

print(np.mean(np.abs(model.predict(X[400:]) - y[400:])))
```

Esse modelo é um **Generalized Additive Model** construído com uma LUT por recurso e saída. Interações devem ser criadas explicitamente como novas features.

## HDC

```python
import numpy as np
from addernet import AdderNetHDC

X = np.array([[0, 0], [0, 1], [9, 10], [10, 9]], dtype=np.float64)
y = np.array([0, 0, 1, 1], dtype=np.int32)

model = AdderNetHDC(n_vars=2, n_classes=2, hv_dim=1025, seed=7)
model.train(X, y, n_iter=5, patience=0)
print(model.predict_batch(X))
```

Dimensões não múltiplas de 64 são suportadas com bounds checks no caminho AVX.

## Attention correta

```python
import numpy as np
from addernet import AdderAttention

Q = np.random.randn(2, 3, 5)
K = np.random.randn(2, 4, 5)
V = np.random.randn(2, 4, 6)

attn = AdderAttention(normalize=True)
output = attn(Q, K, V)  # shape: (2, 3, 6)
scores = attn.scores(Q, K)
```

A classe é uma operação **sem estado** baseada em distância L1 negativa. Ela não possui `fit()` nem `predict()`.

## Componentes

| Componente | Uso |
|---|---|
| `AdderNetLayer` | função escalar quantizada por LUT nativa |
| `ReferenceAdderNetLayer` | versão legível em NumPy |
| `UniformQuantizer` | quantização explícita de entradas contínuas |
| `AdderNetAdditiveRegressor` | regressão multivariada aditiva |
| `AdderNetHDC` | classificação multivariada com hipervetores |
| `AdderCluster` | ensemble de LUTs |
| `AdderBoost` | boosting aditivo |
| `AdderAttention` | pooling por distância L1 |

## Correções desta edição

- auto-build CPU realmente compila e verifica as duas bibliotecas antes do import;
- `addernet-build` retorna código de sucesso correto;
- compilação portátil, sem forçar AVX2 em toda CPU;
- lookup de bibliotecas Linux/macOS/Windows corrigido nos bindings;
- overflow no batch AVX para `hv_dim` não múltiplo de 64 corrigido;
- serialização HDC preserva dimensionalidade dinâmica;
- acesso ao `codebook` usa layout correto;
- limitação de threads evita criação excessiva;
- entradas vazias, incompatíveis, `NaN` e infinito são validadas;
- treino com uma amostra ou entradas duplicadas é seguro;
- documentação de `AdderAttention` sincronizada com a implementação;
- suíte lenta foi movida para `benchmarks/`, mantendo `pytest` rápido;
- implementação Python de referência adicionada;
- regressor multivariado aditivo e quantizador adicionados;
- CLI `addernet-selftest` adicionada.

## Testes

```bash
python -m pytest -q
python benchmarks/validation_suite.py  # validação mais longa
```

## Estrutura

```text
addernet/
  addernet.py          binding da LUT nativa
  addernet_hdc.py      binding HDC
  reference.py         implementação pedagógica
  vector.py            regressão multivariada aditiva
  attention.py         attention L1
  cluster.py           ensemble
  boost.py             boosting
  build_ext.py         compilação CPU portátil
  src/                 fontes C incluídos no wheel
src/                    fontes C do repositório
examples/               exemplos executáveis
tests/                  testes rápidos
benchmarks/             validações longas
APOSTILA_ADDERNET.html   curso interativo completo
```

## Limitações conhecidas

- A LUT nativa recebe uma variável escalar por camada e converte a entrada para inteiro.
- O regressor multivariado é aditivo; não descobre interações complexas sem features extras.
- HDC troca parte da precisão por memória associativa e inferência barata.
- O backend CUDA requer hardware e toolchain compatíveis; esta edição foi validada principalmente em CPU.
- Nenhuma suíte de testes prova ausência absoluta de bugs em toda plataforma.

## Licença

Apache-2.0. Projeto original de Pedro Henrique Batista Silva; esta edição preserva a licença e adiciona correções, material didático e utilitários de aprendizagem.
