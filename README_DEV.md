# Desenvolvimento

## Ambiente

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Build nativo

```bash
addernet-build
# ou
python -m addernet.build_ext
```

Variáveis:

- `ADDERNET_AUTOBUILD=0`: não compilar durante import.
- `ADDERNET_NATIVE=1`: usar `-march=native` em x86-64.
- `ADDERNET_OPENMP=0`: compilar sem OpenMP.
- `ADDERNET_VERBOSE=0`: reduzir logs.

## Filosofia de contribuição

1. Um erro C nunca deve derrubar o interpretador por uma entrada Python válida.
2. Toda alteração de formato binário precisa de roundtrip de save/load.
3. Otimização SIMD deve ser comparada com o caminho escalar.
4. Documentação e assinatura real da API devem permanecer sincronizadas.
5. Benchmarks devem separar custo de treino, inferência, pré-processamento e qualidade.
