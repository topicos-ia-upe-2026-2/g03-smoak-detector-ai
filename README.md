# Detecção automática de fogo e fumaça com YOLOv7

Projeto aplicado da disciplina **Tópicos Avançados em Inteligência Artificial — 2026.2**.

**Docente:** Marcondes Ricarte

| Integrantes |
|---|
| Grazielly Lauane, Lívia Maria e Lucas Terêncio|

## Cliente, problema e beneficiário

O cliente de referência é o **Núcleo Verde**, uma empresa fictícia de monitoramento ambiental responsável por examinar imagens de câmeras instaladas em áreas verdes e pontos de observação de um município.

A análise manual de vários pontos de monitoramento exige atenção contínua. Nesse processo, sinais visuais de fumaça ou chamas podem passar despercebidos ou demorar a ser revisados. O projeto propõe um detector automático para destacar imagens suspeitas, indicar onde aparecem possíveis ocorrências e organizar os registros que devem ser revisados.

O usuário principal é o operador ou analista de monitoramento que trabalha em um computador comum. O beneficiário direto é a equipe responsável pela triagem das imagens, que poderá priorizar casos suspeitos e encaminhá-los para verificação por pessoas capacitadas. Indiretamente, o sistema pode apoiar a proteção de áreas verdes e da população atendida pelo monitoramento municipal.

O fluxo operacional é uma hipótese de projeto, pois o Núcleo Verde não representa uma organização real entrevistada. A adequação desse fluxo deverá ser verificada posteriormente por teste de tarefa e coleta de feedback de um representante do público-alvo.

## Solução e escopo

O protótipo receberá uma imagem em formato JPG ou PNG e executará inferência com um modelo YOLOv7 adaptado ao conjunto D-Fire. Após o pré-processamento, as detecções passarão por limiar de confiança e supressão de não máximos (NMS).

Como saída, o sistema deverá produzir:

- uma imagem anotada com caixas delimitadoras e os rótulos `fogo` e/ou `fumaça`;
- uma mensagem de suspeita visual;
- uma tabela ou arquivo CSV com nome do arquivo, classe detectada, confiança e coordenadas de cada detecção;
- uma interface web local desenvolvida com Gradio para envio e análise das imagens.

A decisão final continuará sob responsabilidade do operador, que deverá revisar a imagem original e seguir o protocolo local. O sistema não realizará ações automáticas de emergência.

### Contemplado

- fine-tuning do YOLOv7 para as classes fogo e fumaça;
- inferência em imagens estáticas;
- interface web local com Gradio;
- exportação dos resultados em CSV;
- cálculo de métricas e comparação com um baseline;
- demonstração com imagens que não participaram do treinamento;
- análise opcional de vídeos MP4 curtos por amostragem de quadros, se houver tempo e desempenho suficiente.

### Fora do escopo

- sistema homologado ou certificado de detecção de incêndios;
- acionamento automático de bombeiros ou integração com centrais de alarme;
- sensores físicos, câmeras ao vivo via RTSP ou operação autônoma 24/7;
- aplicativo móvel, geolocalização do foco ou estimativa da área queimada;
- rastreamento temporal e alertas por push ou SMS.

## Artigo-base

- **Título:** *YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors*
- **Autores:** Chien-Yao Wang, Alexey Bochkovskiy e Hong-Yuan Mark Liao
- **Ano:** 2023, com versão preliminar publicada no arXiv em 2022
- **DOI/URL:** [10.1109/CVPR52729.2023.00721](https://doi.org/10.1109/CVPR52729.2023.00721) — [arXiv:2207.02696](https://arxiv.org/abs/2207.02696)
- **Componentes utilizados/adaptados:** arquitetura YOLOv7 padrão da família P5, blocos E-ELAN já presentes no modelo, estratégia de reparametrização implementada no código oficial e procedimento de fine-tuning. A cabeça de detecção será ajustada para duas classes. A cabeça auxiliar P6 e o método de rótulos *coarse-to-fine* não fazem parte do escopo inicial.

O grupo aplicará e avaliará a arquitetura na detecção de fogo e fumaça, sem reivindicar a criação de um novo método. Os resultados do artigo no MS COCO são apenas referência e não serão tratados como resultados deste projeto.

## Modelo pré-treinado

- **Nome e versão:** YOLOv7 padrão, família P5, com entrada de 640 × 640 pixels e configuração `cfg/training/yolov7.yaml`.
- **Origem dos pesos:** arquivo `yolov7_training.pt`, disponibilizado na release v0.1 do [repositório oficial do YOLOv7](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1). O arquivo `yolov7.pt` será usado apenas como referência qualitativa de inferência nas classes do MS COCO.
- **Licença:** código oficial sob GPL-3.0. Não foi identificada uma licença separada específica para os pesos; por cautela, o projeto preservará créditos, origem e condições de distribuição associadas ao repositório.
- **Tarefa original:** detecção de objetos nas 80 classes do MS COCO.
- **Relação com o artigo-base:** trata-se da implementação oficial vinculada ao artigo. O projeto utilizará seus componentes arquiteturais e o fluxo oficial de transferência de aprendizado, sem reimplementar a rede.

A adaptação consistirá em ajustar a cabeça de detecção para `nc: 2` e atualizar os parâmetros por fine-tuning. Se for necessário manter uma configuração própria, será criada e versionada uma cópia de `yolov7.yaml`, com todas as alterações documentadas.

## Conjunto de dados-base

- **Fonte e acesso:** [D-Fire: an image dataset for fire and smoke detection](https://github.com/gaia-solutions-on-demand/DFireDataset), desenvolvido pela GAIA. O repositório fornece links oficiais do OneDrive e indica um espelho no Kaggle.
- **Licença/autorização:** o repositório declara CC0 1.0 Universal. Imagens provenientes de fontes externas continuam sujeitas aos termos dos provedores originais e podem exigir créditos adicionais.
- **Formato e volume:** 21.527 imagens e 26.557 caixas delimitadoras em formato YOLO, com coordenadas normalizadas. O conjunto possui 14.692 caixas de fogo e 11.865 caixas de fumaça.
- **Rótulos e adequação ao problema:** duas classes, fogo e fumaça, distribuídas entre imagens somente com fogo, somente com fumaça, com ambos e sem nenhuma ocorrência. O mapeamento numérico das classes será confirmado diretamente nos arquivos baixados antes do treinamento.

| Tipo de imagem ou anotação | Quantidade |
|---|---:|
| Somente fogo | 1.164 imagens |
| Somente fumaça | 5.867 imagens |
| Fogo e fumaça | 4.658 imagens |
| Sem fogo ou fumaça | 9.838 imagens |
| **Total** | **21.527 imagens** |
| Caixas de fogo | 14.692 |
| Caixas de fumaça | 11.865 |
| **Total de caixas** | **26.557** |

Arquivos grandes de dados e pesos não devem ser versionados no repositório. Devem ser mantidos apenas os scripts, o manifesto, os links e as instruções de download.

## Plano de adaptação e avaliação

- **O que será ajustado:** cabeça de detecção para duas classes, parâmetros do modelo por fine-tuning, arquivo `data/dfire.yaml`, limiar de confiança e hiperparâmetros baseados em `data/hyp.scratch.custom.yaml`. Serão testadas augmentations moderadas, como variações de brilho e saturação e espelhamento horizontal quando apropriado.
- **Recursos computacionais:** Google Colab com GPU, preferencialmente NVIDIA T4 quando disponível, com Kaggle Notebook como alternativa. O ambiente utilizará Python, PyTorch e o repositório oficial do YOLOv7. Versões, seeds, comandos, checkpoints e correções de compatibilidade serão registrados.
- **Baseline:** YOLOv7 com parte do backbone congelada e cabeça de duas classes treinada sobre a mesma divisão de dados. O fine-tuning principal utilizará mais camadas liberadas. A inferência com o modelo treinado apenas no COCO será somente uma referência qualitativa.
- **Métricas:** mAP@0.5, mAP@0.5:0.95, precisão e recall por classe; recall específico de fumaça; taxa de imagens negativas que geram alerta falso; tempo médio por imagem; FPS estimado; e análise qualitativa de pelo menos 20 erros.
- **Riscos:** limitações de GPU, dependências antigas do YOLOv7, classes ou caixas incorretas, duplicatas entre partições, diferença de domínio entre o D-Fire e câmeras locais, falsos positivos por neblina, nuvens, reflexos ou sol e falsos negativos em focos distantes ou fumaça pouco intensa.
- **Divisão das atividades:** Grazielly Lauane conduz problema, beneficiário, documentação, README e validação com o usuário; Lívia Maria conduz download, auditoria, partições, `data/dfire.yaml`, pré-processamento e documentação dos dados; Lucas Terêncio conduz ambiente do YOLOv7, baseline, fine-tuning, métricas, checkpoints e custo computacional. Interface Gradio, integração, testes ponta a ponta, análise de falhas, apresentação e defesa técnica serão atividades conjuntas.

### Experimentos previstos

| Execução | Configuração inicial | Finalidade |
|---|---|---|
| Teste técnico | Subconjunto estratificado de 3–5 mil imagens, 640 px, batch 8 e 5–10 épocas | Verificar ambiente, leitura de rótulos, cálculo da perda e salvamento de pesos |
| Baseline | YOLOv7 com backbone parcialmente congelado e cabeça de duas classes | Medir o efeito de atualizar uma parte menor da rede |
| Fine-tuning principal | Camadas liberadas, conjunto completo se a GPU permitir, 30–50 épocas e batch 8 ou 16 | Avaliar precisão, recall, mAP, erros e custo por imagem |
| Variante leve opcional | YOLOv7-tiny adaptado | Investigar a relação entre velocidade e precisão |

Os valores acima são parâmetros iniciais de planejamento e poderão ser ajustados à memória da GPU e ao progresso do projeto.

## Como executar

> A implementação ainda será integrada ao repositório. Os comandos abaixo descrevem o fluxo planejado e deverão ser atualizados se a estrutura final de arquivos for diferente.

```bash
# 1. Clonar a implementação oficial e criar o ambiente
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 2. Baixar o D-Fire pelos links oficiais e organizar train/val/test.
# Criar data/dfire.yaml com nc: 2 e a ordem de classes confirmada nos metadados.
# Baixar yolov7_training.pt da release oficial v0.1.

# 3. Teste técnico inicial
python train.py --workers 4 --device 0 --batch-size 8 --data data/dfire.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --name dfire-teste --hyp data/hyp.scratch.custom.yaml --epochs 5

# 4. Fine-tuning principal
python train.py --workers 4 --device 0 --batch-size 8 --data data/dfire.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --name dfire-yolov7 --hyp data/hyp.scratch.custom.yaml --epochs 50

# 5. Avaliação no conjunto de teste
python test.py --data data/dfire.yaml --img-size 640 --batch-size 8 --device 0 --weights runs/train/dfire-yolov7/weights/best.pt --name dfire-teste-final

# 6. Inferência em uma imagem ou pasta
python detect.py --weights runs/train/dfire-yolov7/weights/best.pt --conf 0.25 --img-size 640 --source caminho/para/imagens

# 7. Interface local, após sua implementação
# python src/app.py
```

## Resultados

O projeto ainda se encontra na fase de proposta e, portanto, não possui resultados experimentais próprios. A hipótese inicial é buscar **mAP@0.5 ≥ 0,70** e **recall de fumaça ≥ 0,80** em um conjunto de teste isolado. Essas metas não são critérios obrigatórios de aprovação: resultados inferiores também deverão ser relatados e discutidos.

Os resultados finais deverão incluir:

- métricas por classe e agregadas para o baseline e o fine-tuning principal;
- limiar de confiança escolhido somente com o conjunto de validação;
- tempo médio por imagem e FPS, acompanhados da identificação do hardware;
- taxa de falsos alertas em imagens negativas;
- exemplos de acertos, falsos positivos e falsos negativos;
- análise de pelo menos 20 erros, agrupados por iluminação, neblina ou nuvens, reflexos, oclusão e distância;
- teste de generalização com imagens autorizadas de um cenário diferente do D-Fire.

## Limitações e uso responsável

Este é um **protótipo acadêmico de apoio à triagem visual**. Uma detecção indica apenas suspeita de fogo ou fumaça e deve ser revisada por uma pessoa capacitada. O sistema não confirma a existência de um incêndio e não substitui sensores certificados, alarmes físicos, equipes de emergência ou decisões de segurança.

O desempenho pode diminuir em cenários diferentes dos dados de treinamento e diante de fumaça pouco intensa, focos distantes, baixa iluminação, neblina, nuvens, reflexos, oclusões ou imagens degradadas. Falsos positivos e falsos negativos são esperados. Por isso, o protótipo não deve operar de modo autônomo em ambientes críticos nem acionar serviços de emergência.

Antes da distribuição dos dados ou de imagens de demonstração, devem ser verificadas as licenças e autorizações aplicáveis. A coleção D-Fire declara CC0, mas materiais incorporados de fontes externas podem conservar condições próprias. O código e os pesos também devem manter sua origem e seus créditos documentados.

## Referências

1. WANG, C.-Y.; BOCHKOVSKIY, A.; LIAO, H.-Y. M. [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, p. 7464–7475, 2023. DOI: [10.1109/CVPR52729.2023.00721](https://doi.org/10.1109/CVPR52729.2023.00721).
2. WONGKINYIU. [Repositório oficial do YOLOv7](https://github.com/WongKinYiu/yolov7). Implementação, treinamento, pesos e licença GPL-3.0.
3. GAIA. [D-Fire: an image dataset for fire and smoke detection](https://github.com/gaia-solutions-on-demand/DFireDataset). Descrição, totais, formato e links de download.
4. GAIA. [Licença do D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset/blob/master/LICENSE). CC0 1.0 Universal e observações sobre direitos de terceiros.
5. VENÂNCIO, P. V. A. B.; LISBOA, A. C.; BARBOSA, A. V. [An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices](https://doi.org/10.1007/s00521-022-07467-z). *Neural Computing and Applications*, 2022.

