# Registro de experimentos

> Documente cada execução relevante: data, configuração, hiperparâmetros, dados usados,
> métricas obtidas e observações. Isso fundamenta as seções de resultados da E2, E3 e E4.

## Baseline

Executado em GPU com o subconjunto inicial descrito abaixo. A equivalência das partições do espelho Kaggle com as listas oficiais do D-Fire ainda precisa ser confirmada; portanto, as métricas são evidência inicial da E2, não resultado final do projeto. Protocolo: [e2_dados_baseline.md](e2_dados_baseline.md).

### Validação técnica no Colab

- Ambiente: Google Colab, Tesla T4, Python 3.13 e PyTorch 2.14.1+cu130.
- YOLOv7 oficial: commit `a207844b1ce82d204ab36d87d496728d3d2348e7`.
- Fonte testada: espelho Kaggle `sayedgamal99/smoke-fire-detection-yolo`; a equivalência das partições com as listas oficiais ainda precisa ser verificada.
- Auditoria inicial: 21.527 imagens e 397 caixas inválidas (379 cruzavam a borda e 18 tinham área zero).
- Preparação: 379 caixas recortadas aos limites, com no mínimo 85,06% da área original preservada; 18 caixas sem área removidas; tolerância numérica de borda de `1e-9`.
- Auditoria final: zero erros, 11.854 caixas de fumaça e 14.685 de fogo, total de 26.539 caixas válidas.
- Compatibilidade: o pin `numpy<1.24` do YOLOv7 não compila no Python 3.13. O teste manteve Torch, torchvision, NumPy e Protobuf do Colab e instalou as demais dependências pela lista versionada `requirements-colab.txt`, derivada do arquivo oficial. O rastreamento W&B foi desativado para não exigir conta externa, e `np.trapz` foi mapeado para `np.trapezoid` em `src/yolov7_compat/sitecustomize.py`, pois a função antiga não existe no NumPy atual.
- Essas contagens são resultados de preparação de dados. Não são métricas de desempenho do detector.
- Teste de fumaça do pipeline: 96 imagens de treino, 24 de validação e 24 de teste, uma época, 320 px, batch 4 e 50 camadas congeladas. O processo gerou `best.pt`, executou `test.py` e terminou com código 0. As métricas muito baixas dessa amostra curta não representam o baseline e não devem ser comparadas como resultado científico.
- Subconjunto de treino da E2: 3.000 imagens, `seed=42`, estratificação por presença de classes: 1.372 sem ocorrência, 815 somente com fumaça, 163 somente com fogo e 650 com ambas. Validação completa com 3.099 imagens; teste completo com 4.306 imagens.
- Baseline comparável: pesos oficiais `yolov7_training.pt`, `freeze=50`, 3 épocas, 640 px, batch 8. Treino concluído em 620,4 s e teste em 96,3 s.
- Baseline no teste: P=0,399, R=0,378, mAP@0.5=0,322 e mAP@0.5:0.95=0,128. Fumaça: P=0,455, R=0,439, mAP@0.5=0,371 e mAP@0.5:0.95=0,162. Fogo: P=0,344, R=0,318, mAP@0.5=0,273 e mAP@0.5:0.95=0,0946.
- Artefatos temporários do Colab: `/content/e2_artifacts/e2_3000_baseline`; checkpoints e arquivos grandes não são versionados no Git.

Ao executar, registrar data, commits do projeto e YOLOv7, origem e versão dos dados, confirmação do ID das classes, contagens por partição, ambiente/GPU, duração, hiperparâmetros, precisão, recall, mAP@0.5, mAP@0.5:0.95 gerais e por classe, caminho dos logs e problemas encontrados.

## Versões ajustadas (fine-tuning)

Configuração inicial fixada para a E2: 3.000 imagens de treino selecionadas de forma estratificada com `seed=42`, validação e teste completos, 3 épocas, 640 px, batch 8 e nenhuma camada congelada. Comparar com um baseline nas mesmas condições usando `freeze=50`.

| Data | Descrição | Hiperparâmetros | Dados | Métricas | Observações |
|---|---|---|---|---|---|
| 22/11/2026 | Fine-tuning inicial YOLOv7 P5 | pesos `yolov7_training.pt`, 3 épocas, 640 px, batch 8, `freeze=0`, `hyp.scratch.custom.yaml` | 3.000 treino; 3.099 validação; 4.306 teste | teste: P=0,317; R=0,284; mAP@0.5=0,204; mAP@0.5:0.95=0,0694. Fumaça: mAP@0.5=0,226; fogo: mAP@0.5=0,183 | Treino 771,5 s; teste 88,9 s; execução concluída e `best.pt` gerado. Resultado inicial inferior ao baseline congelado. |

## Resultados iniciais e obstáculos

- O pipeline completo foi comprovado: seleção determinística, treino, validação, geração de `best.pt` e avaliação no teste terminaram com código 0 na Tesla T4.
- Liberar todas as camadas por somente 3 épocas reduziu o mAP@0.5 de 0,322 para 0,204 e o recall de 0,378 para 0,284. Isso é um resultado medido, não uma estimativa. A hipótese mais provável é que o `hyp.scratch.custom.yaml` e seu aquecimento, adequado a treinos mais longos, tornem três épocas insuficientes para estabilizar todas as camadas.
- O fine-tuning custou 771,5 s contra 620,4 s do baseline, aumento de aproximadamente 24% no tempo de treino, além de não melhorar as métricas nesse horizonte curto.
- Risco de dados: falta confrontar as partições do espelho Kaggle com as listas oficiais e concluir a confirmação visual do mapeamento numérico das classes.
- Plano para a E3: confirmar as listas oficiais e a ordem das classes; preservar o teste; ajustar taxa de aprendizado e aquecimento para transferência; experimentar liberação gradual de camadas; aumentar o número de épocas com early stopping baseado na validação; repetir com pelo menos três seeds; medir falsos alertas em negativos e analisar exemplos de falso positivo e falso negativo.
- Artefatos temporários do Colab: `/content/e2_artifacts/e2_3000_finetune`. Antes de encerrar a sessão, checkpoints podem ser copiados para armazenamento persistente respeitando a licença; eles permanecem fora do Git.
