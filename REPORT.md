# Relatório resumido — auditoria e expansão da AdderNet

## Resultado

A biblioteca `addernet` 1.5.0 foi instalada e usada como base para uma versão de
desenvolvimento organizada como `1.6.0.dev0`. O pacote melhorado foi instalado
em modo editável neste ambiente, compilado, testado e também empacotado em um
wheel Linux x86-64 para Python 3.13.

## Principais problemas encontrados

1. **Camada escalar apenas:** `AdderNetLayer` aceitava somente um valor de entrada
e produzia somente um valor de saída. O regressor aditivo existente aceitava
várias features, mas não modelava interações reais entre `x` e `y`.
2. **Perda silenciosa de dados:** a interpolação nativa copiava no máximo 256
amostras para um buffer fixo. Amostras posteriores eram ignoradas.
3. **Treinamento desnecessariamente iterativo:** tarefas de LUT podiam exigir
milhares de épocas mesmo quando a média por célula e a interpolação fornecem uma
solução direta.
4. **OpenMP contraproducente:** o loop de inferência é pequeno e limitado por
memória; iniciar muitas threads adicionava overhead considerável no host de
teste.
5. **Carregamento binário frágil:** tamanho, intervalo, learning rate, arquivo
truncado e valores NaN não eram completamente validados.
6. **API de ciclo de vida incompleta:** não havia `close()`, context manager nem
proteção clara contra uso após liberar o ponteiro.
7. **Serialização pouco portátil:** o formato nativo dependia da representação
binária da plataforma e não tinha versão explícita.
8. **Quantização ambígua:** um vetor 1-D podia significar várias amostras de uma
feature ou uma única amostra de várias features.
9. **Modelos de alto nível lentos por padrão:** boost, cluster e regressor aditivo
usavam o treinamento iterativo mesmo quando o ajuste direto era suficiente.

## Correções e funcionalidades implementadas

### Núcleo C

- Remoção do limite fixo de 256 amostras.
- Ordenação dinâmica com `qsort` e agregação das entradas duplicadas pela média.
- Nova função `an_fit_direct`, com complexidade aproximada `O(n + intervalo)`.
- Atualização incremental com `blend` para lotes sucessivos.
- Nova função segura para substituir a tabela (`an_set_offset`).
- Validação de finitude, limites de inteiro, tamanho, épocas e learning rate.
- Carregamento endurecido contra cabeçalhos inválidos, arquivos truncados e NaN.
- Verificação de todas as escritas e liberações alinhadas portáveis.
- OpenMP passou a ser opcional via `ADDERNET_OPENMP=1`, não mais padrão.

### API escalar

- `fit`, `partial_fit`, `close`, context manager e validação de estado.
- `train` agora retorna `self` e mantém compatibilidade com o algoritmo antigo.
- Salvamento nativo atômico.
- `save_portable` e `load_portable` em formato NumPy versionado.
- `set_offset_table`, `get_params` e checagem dos códigos de retorno nativos.

### Múltiplas entradas e saídas

Foi criada `AdderNetMultiInputLayer`, também exportada como
`AdderNetRegressor`.

Ela oferece:

- N entradas e uma ou várias saídas.
- Chamada direta como `model.predict(x, y)`.
- Entrada em lote com shape `(amostras, features)`.
- LUTs aditivas por feature.
- LUTs conjuntas para pares de features, capturando interações como `x * y`, XOR
e outras funções que não podem ser separadas em `f(x) + g(y)`.
- Seleção automática, total, desativada ou manual das interações.
- `predict_quantized`, `score`, `explain`, `memory_bytes_`, `save` e `load`.
- Índices conjuntos produzidos com bit shift e OR para bins potência de dois.

Também foi adicionada `AdderNetClassifier`, com labels arbitrários,
`decision_function`, `predict_proba`, `score` e serialização.

### Organização

- Novos modelos em `addernet/models/`.
- Pré-processamento em `addernet/preprocessing/`.
- Testes em `tests/`.
- Benchmarks em `benchmarks/`.
- Exemplos em `examples/`.
- Documentação de arquitetura em `docs/`.
- `pyproject.toml`, wheel com tag de plataforma, changelog e manifesto de fontes.

## Testes executados

- **21 testes automatizados: 21 aprovados.**
- Autoteste original: **PASS**.
- HDC: acurácia de treinamento **1.000** no autoteste.
- Round-trip de modelos escalares, multivariados, classificadores e regressor
aditivo.
- Caso com mais de 256 amostras validando que o truncamento foi eliminado.
- Rejeição de arquivo nativo com cabeçalho malicioso.
- Caso `f(x, y)` com duas entradas e três saídas reproduzido com erro numérico
praticamente zero em uma grade completamente observada.
- Wheel instalado em diretório isolado e importado com sucesso.

## Benchmark no host de teste

Os tempos variam conforme CPU, compilador e carga do sistema. Estes números são
comparações locais, não garantias universais.

| Medida | 1.5.0 original | 1.6.0.dev0 | Resultado |
|---|---:|---:|---:|
| Treino escalar iterativo, 2.000 amostras | 5,798 ms | 4,983 ms | 1,16× mais rápido |
| Ajuste escalar direto, 2.000 amostras | — | 0,283 ms | 20,45× vs. treino original |
| Inferência de 1 milhão de valores | 5,393 ms | 2,304 ms | 2,34× mais rápido |
| Throughput escalar | 185,4 milhões/s | 434,0 milhões/s | +134% |

Benchmark ampliado:

- 20.000 amostras, ajuste direto: **14,44×** mais rápido que o iterativo da nova versão.
- Modelo de duas entradas com 50.000 amostras: **7,70 ms** de ajuste.
- RMSE de treino desse benchmark multivariado: **0,0142**.

Os JSONs brutos estão em `benchmarks/original_1.5.0.json`,
`benchmarks/enhanced_1.6.0.json` e `benchmarks/full_benchmark.json`.

## Limitações restantes

- LUTs de interação consomem memória proporcional a `bins²`; por isso existe
`max_interactions`.
- A implementação nova cobre interações de segunda ordem. Tensores conjuntos de
ordem maior cresceriam exponencialmente.
- `predict_quantized` mantém o núcleo por lookup, bit operations e soma. A rota
com entradas float ainda precisa quantizar os valores e essa etapa usa aritmética
convencional.
- CUDA e o núcleo HDC foram preservados e verificados pelo autoteste, mas não
foram reescritos nem submetidos a profiling profundo nesta revisão.
- O wheel entregue é específico para Linux x86-64 e CPython 3.13. Em outra
plataforma, deve-se recompilar pelo código-fonte.
- As mudanças não foram enviadas ao repositório remoto; o ambiente não forneceu
acesso Git autenticado pelo terminal.
