# CHANGELOG

## 2026-03-12 — melhorias: seleção de dataset e correções

Adicionado/Modificado:

- `train_models.py`
  - Adicionada interface de linha de comando `--dataset {abilene,rnp}` e `--dry-run`.
  - Agora é possível alternar entre os datasets sem editar o código.

- `src/model_launcher.py`
  - `train()` aceita parâmetro `dataset` e seleciona automaticamente a pasta de traffic matrices correspondente.
  - Instancia o modelo somente após carregar os dados, ajustando dinamicamente os parâmetros de entrada (`node_input_dim`, `edge_input_dim`).

- `src/utils/train_util.py`
  - Correção na construção dos caminhos dos arquivos de traffic matrices; tenta o caminho do dataset Abilene quando aplicável.
  - Passa a flag `abilene=True` ao chamar `DataUtil.get_node_loads` para alinhar o formato dos TMs do Abilene.

- `README.md`
  - Nova seção com instruções para alternar entre datasets e exemplos de uso de `--dry-run`.

Motivação:

- Facilitar experimentação entre os dois datasets disponíveis (Abilene e RNP) sem necessidade de edição manual do código.
- Corrigir erros de path e incompatibilidades de dimensões observadas ao usar o dataset Abilene.

Notas:

- Se desejar que a seleção de dataset seja persistente em configurações, considere adicionar uma propriedade `dataset` na classe `Config`.
- Recomenda-se revisar `src/constants.py` caso queira alterar os caminhos padrões dos datasets ou TMs.
