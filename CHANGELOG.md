# Changelog

## 1.5.0 — Official Release

### Correções

- corrigido auto-build que chamava o gerenciador CUDA sem produzir as bibliotecas CPU;
- build agora verifica os arquivos produzidos antes de importar bindings;
- `addernet-build` retorna código 0 em sucesso;
- compilação CPU portátil por padrão; otimizações nativas são opt-in;
- lookup de bibliotecas atualizado para Linux, macOS e nomes Windows;
- documentação de `AdderAttention` corrigida para a API sem estado real;
- validações longas removidas da coleta padrão do pytest;
- preservadas correções anteriores de overflow AVX, dimensão HDC dinâmica, codebook, threads e entradas degeneradas.

### Melhorias

- `ReferenceAdderNetLayer`: implementação NumPy transparente;
- `UniformQuantizer`: quantização explícita e persistível;
- `AdderNetAdditiveRegressor`: regressão multivariada por backfitting;
- `addernet-selftest`: diagnóstico de instalação;
- novos exemplos e testes;
- README técnico reescrito com afirmações mais precisas;
- apostila interativa completa em um único HTML.
