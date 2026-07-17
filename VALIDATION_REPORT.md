# Relatório de validação — AdderNet Learning Edition 1.5.0.post1

Data: 2026-07-16  
Ambiente principal: Linux x86-64, Python 3.13.5, CPU, sem CUDA.

## Resultados

### Suíte rápida

```text
24 passed in 4.22s
```

Cobertura prática incluída:

- criação, treino, inferência e serialização de `AdderNetLayer`;
- amostra única e entradas duplicadas;
- validação de arrays vazios, shapes, labels, NaN e infinito;
- HDC com `hv_dim` 511, 512, 1025 e 2500;
- comparação entre batch comum, caminho AVX e multithreading;
- limite seguro ao solicitar 1.000 threads;
- cache HDC;
- save/load HDC com dimensão dinâmica;
- layout e classificação do codebook;
- API e validação de `AdderAttention`;
- `AdderBoost`;
- detector CUDA e fallbacks por mocks;
- implementação NumPy de referência;
- quantização com coluna constante;
- regressor multivariado e roundtrip de diretório.

### Auto-build do código-fonte

Os binários nativos foram removidos e o import recompilou:

```text
addernet/libaddernet.so
addernet/libaddernet_hdc.so
version=1.5.0.post1
backend=SCALAR
```

### Wheel em ambiente limpo

Artefato:

```text
addernet-1.5.0.post1-cp313-cp313-linux_x86_64.whl
```

Saída do self-test:

```text
backend=SCALAR
layer_mae=0.200000
hdc_training_accuracy=1.000
attention_shape=(1, 1, 3)
save_load_roundtrip=True
SELFTEST: PASS
```

Teste adicional do regressor multivariado:

```text
vector_mae=0.038074229691887125
reference_pred_7=14.0
```

### Pacote-fonte instalado em diretório isolado

Artefato:

```text
addernet-1.5.0.post1.tar.gz
```

O pacote foi construído e instalado em um target separado. O mesmo self-test passou.

### Apostila HTML

- parse HTML concluído;
- JavaScript extraído e validado com `node --check`;
- 15 módulos numerados, introdução e glossário;
- sete laboratórios interativos;
- mais de 40 atividades entre questões, código, práticas e projeto final;
- nenhum recurso de rede externo;
- progresso salvo via `localStorage`.

## Limitações da validação

- Nenhuma GPU CUDA estava disponível para executar kernels reais.
- O wheel fornecido é específico para Python 3.13/Linux x86-64; outras plataformas devem usar o pacote-fonte.
- Testes não provam ausência total de defeitos em toda plataforma, compilador ou entrada.
- O benchmark longo em `benchmarks/validation_suite.py` não faz parte da suíte rápida e deve ser executado separadamente.
