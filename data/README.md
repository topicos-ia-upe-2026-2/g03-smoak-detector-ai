# D-Fire para a E2

Fonte oficial: https://github.com/gaia-solutions-on-demand/DFireDataset

O link **D-Fire dataset (only images and labels)** do fornecedor aponta para `D-Fire.zip` (2,83 GB). O link **Training, validation and test sets** aponta para arquivos `.txt` com listas de nomes, inclusive `dfire_train.txt`, `dfire_test.txt` e arquivos de validação cruzada. Não contém pastas prontas com imagens. O OneDrive exibiu um aviso de malware ao tentar baixar o ZIP neste ambiente em 22/11/2026; não foi possível verificar ou baixar seu conteúdo. Como alternativa, o próprio fornecedor indica um [espelho Kaggle pronto para uso](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo); confirme estrutura, contagens e procedência antes do treino.

Organize os dados no Google Drive ou em disco local, fora do Git, com esta estrutura:

```text
dfire/
  train/images/  train/labels/
  val/images/    val/labels/
  test/images/   test/labels/
```

Se o pacote usar outros nomes de pasta, ajuste apenas as pastas de partição, não os arquivos. Se usar o ZIP oficial, aplique as listas de nomes oficiais para gerar as partições e documente a regra usada para selecionar a validação. Se usar o espelho Kaggle, registre que a divisão veio do espelho e confira se coincide com as listas oficiais antes de chamá-la de oficial. Cada imagem precisa de um `.txt` correspondente, vazio quando não há caixa. Não exclua imagens negativas. O script valida integridade, formato YOLO, caixas, rótulos órfãos e duplicatas exatas por SHA-256, sem alterar os originais. `src/clean_dfire.py` cria uma cópia derivada e um `cleaning.csv`: recorta aos limites caixas que preservem pelo menos 80% da área e remove caixas de área zero. Duplicatas perceptuais e quadros vizinhos de um mesmo vídeo exigem revisão visual adicional e ficam como limitação do protocolo.

O README oficial informa as duas classes, mas **não fixa a correspondência numérica dos IDs**. Antes da auditoria, abra amostras anotadas de cada ID e confirme visualmente `0=fumaça, 1=fogo` (ou o inverso). Os argumentos obrigatórios impedem assumir a ordem silenciosamente. Confira também as contagens por classe no relatório gerado; as caixas devem ser compatíveis com os totais oficiais (14.692 de fogo e 11.865 de fumaça), observando que versões ou espelhos podem divergir.

```bash
python -m pip install -r requirements.txt
python src/clean_dfire.py --root /caminho/dfire --output /caminho/dfire_clean
python src/prepare_dfire.py --root /caminho/dfire_clean --class-zero fumaça --class-one fogo --output artifacts/data_audit
```

Saídas: `manifest.csv` com caminho relativo, categoria, dimensões e hash; `summary.json` com contagens e erros; `dfire.yaml` com caminhos absolutos para o YOLOv7. O YAML só é gerado após auditoria sem erros. Gere novamente ao mudar de máquina ou ambiente. Os artefatos ficam fora do Git para evitar caminhos pessoais e arquivos grandes; guarde cópia do resumo e do log da execução para a E2.

Para a execução inicial reduzida da E2, gere uma seleção estratificada de 3.000 imagens. O script preserva validação e teste completos no YAML e registra exatamente quais imagens foram selecionadas:

```bash
python src/create_dfire_subset.py --root /caminho/dfire_clean --output /caminho/dfire_e2_3000 --size 3000 --seed 42
```

Licença: o repositório do D-Fire declara CC0, mas avisa que imagens oriundas de terceiros continuam sujeitas aos termos de origem. Não redistribua imagens antes de verificar esses termos.
