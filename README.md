# E2 — Dados preparados e primeira adaptação

Esta branch reúne somente a segunda entrega do **Smoak Detector AI**: preparação dos dados, baseline comparável, fine-tuning inicial e registro dos primeiros resultados e obstáculos.

## Estrutura atual do projeto

```text
smoak-detector-ia/
├── data/
│   └── README.md                    # origem, licença e organização do D-Fire
├── docs/
│   ├── decisoes.md                  # decisões técnicas do projeto
│   ├── e2_dados_baseline.md         # protocolo reproduzível da E2
│   ├── experimentos.md              # métricas, tempos e obstáculos
│   └── relatorio.md                 # conteúdo-fonte do relatório da entrega
├── notebooks/
│   └── e2_dados_baseline.ipynb      # execução completa no Google Colab
├── output/
│   └── pdf/
│       └── relatorio_e2_smoak_detector.pdf
├── src/
│   ├── yolov7_compat/
│   │   └── sitecustomize.py         # compatibilidade com o Colab atual
│   ├── clean_dfire.py               # limpeza das anotações
│   ├── create_dfire_subset.py       # seleção estratificada de 3.000 imagens
│   ├── prepare_dfire.py             # auditoria e geração do dfire.yaml
│   └── run_baseline.py              # baseline, fine-tuning e avaliação
├── CONTRIBUTING.md
├── requirements.txt                 # dependências diretas do projeto
├── requirements-colab.txt           # dependências adicionais de treinamento
└── README.md                         # resumo da entrega E2
```

Imagens do dataset, pesos e checkpoints não aparecem nessa árvore porque permanecem fora do Git. Eles são obtidos ou gerados durante a execução no Colab.

O pipeline foi executado no Google Colab com GPU Tesla T4. Foram usadas **3.000 imagens de treino**, 3.099 de validação e 4.306 de teste. O baseline congelado obteve **mAP@0.5 de 0,322** e o fine-tuning inicial obteve **0,204**. Esses valores são resultados medidos de uma execução curta de 3 épocas e não representam o treinamento final do projeto.

## O que foi entregue

- pipeline reproduzível de limpeza, auditoria e preparação do D-Fire;
- registro da origem, estrutura e limitações dos dados;
- seleção estratificada e determinística de 3.000 imagens de treino;
- baseline YOLOv7 com parte da rede congelada;
- fine-tuning inicial com todas as camadas liberadas;
- avaliação nas mesmas partições e condições;
- métricas, tempos, limitações e plano de correção para a próxima etapa;
- notebook pronto para execução no Google Colab;
- relatório resumido em PDF.

## Dados

Foi utilizado o conjunto **D-Fire**, com duas classes: fumaça e fogo. A execução testada obteve os arquivos pelo espelho Kaggle indicado pelo fornecedor do dataset. A equivalência exata das partições desse espelho com as listas oficiais ainda precisa ser confirmada.

A auditoria inicial encontrou:

| Indicador | Resultado |
|---|---:|
| Imagens nas três partições | 21.527 |
| Caixas inválidas | 397 |
| Caixas que ultrapassavam os limites | 379 |
| Caixas com área zero | 18 |

A limpeza preservou os arquivos originais e criou uma cópia derivada. As 379 caixas foram recortadas aos limites e as 18 caixas sem área foram removidas. A auditoria final terminou sem erros, com **26.539 caixas válidas**: 11.854 de fumaça e 14.685 de fogo.

Para a execução inicial foi criado um subconjunto de treino com `seed=42` e estratificação por presença de classes:

| Categoria da imagem | Quantidade | Participação no treino |
|---|---:|---:|
| Sem ocorrência | 1.372 | 45,7% |
| Somente fumaça | 815 | 27,2% |
| Somente fogo | 163 | 5,4% |
| Fumaça e fogo | 650 | 21,7% |
| **Total** | **3.000** | **100%** |

A validação completa, com 3.099 imagens, e o teste completo, com 4.306 imagens, foram preservados nas duas execuções.

## Baseline

O baseline permite comparar a primeira adaptação com uma referência nas mesmas condições. A configuração utilizada foi:

| Parâmetro | Valor |
|---|---|
| Modelo | YOLOv7 P5 |
| Pesos iniciais | `yolov7_training.pt` oficial |
| Commit do YOLOv7 | `a207844b1ce82d204ab36d87d496728d3d2348e7` |
| Classes | Fumaça e fogo |
| Camadas congeladas | 50 (`freeze=50`) |
| Imagens de treino | 3.000 |
| Épocas | 3 |
| Resolução | 640 × 640 px |
| Batch | 8 |
| Hiperparâmetros | `hyp.scratch.custom.yaml` |
| Avaliação | Teste preservado com 4.306 imagens |

Resultados do baseline no teste:

| Métrica | Resultado |
|---|---:|
| Precisão | **0,399** |
| Recall | **0,378** |
| mAP@0.5 geral | **0,322** |
| mAP@0.5:0.95 geral | **0,128** |
| mAP@0.5 de fumaça | **0,371** |
| mAP@0.5 de fogo | **0,273** |
| Tempo de treino | **620,4 s** |
| Tempo de teste | **96,3 s** |

## Fine-tuning inicial

O fine-tuning da E2 foi propositalmente curto. Seu objetivo foi comprovar o funcionamento completo do método, e não obter o melhor modelo possível.

A execução manteve os mesmos dados, pesos, resolução, batch, hiperparâmetros e avaliação do baseline, mas liberou todas as camadas (`freeze=0`). O treinamento gerou o checkpoint `best.pt` e concluiu a avaliação com código de saída 0.

Resultados do fine-tuning inicial no teste:

| Métrica | Resultado |
|---|---:|
| Precisão | **0,317** |
| Recall | **0,284** |
| mAP@0.5 geral | **0,204** |
| mAP@0.5:0.95 geral | **0,0694** |
| mAP@0.5 de fumaça | **0,226** |
| mAP@0.5 de fogo | **0,183** |
| Tempo de treino | **771,5 s** |
| Tempo de teste | **88,9 s** |

## Resultados e obstáculos

O pipeline completo foi validado: aquisição, limpeza, auditoria, seleção das 3.000 imagens, treinamento, validação, geração do `best.pt` e teste foram concluídos na Tesla T4.

| Métrica | Baseline | Fine-tuning inicial | Variação |
|---|---:|---:|---:|
| Precisão | 0,399 | 0,317 | -0,082 |
| Recall | 0,378 | 0,284 | -0,094 |
| mAP@0.5 | 0,322 | 0,204 | -0,118 |
| mAP@0.5:0.95 | 0,128 | 0,0694 | -0,0586 |
| Tempo de treino | 620,4 s | 771,5 s | +24,4% |

O fine-tuning inicial ficou abaixo do baseline. Isso não foi tratado como falha de execução. A hipótese registrada é que liberar toda a rede por apenas 3 épocas, usando aquecimento e hiperparâmetros voltados a treinos mais longos, não deu tempo para o modelo estabilizar.

Obstáculos ainda abertos:

- confirmar as partições do espelho Kaggle contra as listas oficiais do D-Fire;
- concluir a confirmação visual da correspondência numérica entre IDs e classes;
- ajustar taxa de aprendizado e aquecimento para transferência curta;
- experimentar liberação gradual de camadas;
- treinar por mais épocas com seleção pela validação e early stopping;
- repetir o experimento com mais seeds;
- analisar falsos positivos, falsos negativos e falsos alertas em imagens negativas.

## Como executar no Google Colab

1. Abra [`notebooks/e2_dados_baseline.ipynb`](notebooks/e2_dados_baseline.ipynb) no Google Colab.
2. Selecione um ambiente com GPU em **Runtime > Change runtime type > GPU**.
3. Execute as células na ordem. O notebook clona o YOLOv7, fixa o commit usado na E2, instala as dependências, prepara os dados e chama os scripts versionados neste repositório.
4. Copie os artefatos desejados para armazenamento persistente antes de encerrar a sessão, pois `/content` é temporário.

As dependências diretas dos scripts estão em [`requirements.txt`](requirements.txt). As dependências adicionais de treinamento estão em [`requirements-colab.txt`](requirements-colab.txt). Torch, torchvision, NumPy e Protobuf são mantidos nas versões fornecidas pelo Colab para preservar a compatibilidade entre Python, CUDA e GPU.

## Onde encontrar cada parte da E2

- **Dados, origem, licença e estrutura:** [`data/README.md`](data/README.md)
- **Limpeza das anotações:** [`src/clean_dfire.py`](src/clean_dfire.py)
- **Auditoria e geração do `dfire.yaml`:** [`src/prepare_dfire.py`](src/prepare_dfire.py)
- **Seleção estratificada das 3.000 imagens:** [`src/create_dfire_subset.py`](src/create_dfire_subset.py)
- **Baseline e fine-tuning inicial:** [`src/run_baseline.py`](src/run_baseline.py)
- **Compatibilidade com o Colab atual:** [`src/yolov7_compat/sitecustomize.py`](src/yolov7_compat/sitecustomize.py)
- **Notebook de execução:** [`notebooks/e2_dados_baseline.ipynb`](notebooks/e2_dados_baseline.ipynb)
- **Protocolo de dados e avaliação:** [`docs/e2_dados_baseline.md`](docs/e2_dados_baseline.md)
- **Métricas e registro das execuções:** [`docs/experimentos.md`](docs/experimentos.md)
- **Relatório resumido:** [`output/pdf/relatorio_e2_smoak_detector.pdf`](output/pdf/relatorio_e2_smoak_detector.pdf)

## Observação sobre arquivos grandes

As imagens do D-Fire, os pesos oficiais e os checkpoints gerados não são versionados no Git por causa do tamanho e das condições de redistribuição. O notebook registra os caminhos dos artefatos temporários e todo o código necessário para reproduzi-los permanece nesta branch.
