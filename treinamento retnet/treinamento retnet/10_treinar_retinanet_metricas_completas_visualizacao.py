# -*- coding: utf-8 -*-
"""
10_treinar_retinanet.py
========================
Treina um RetinaNet (ResNet50-FPN v2, torchvision) nas MESMAS condicoes do seu
YOLOv8m final, para comparar arquiteturas (one-stage anchor-based "classico" x
YOLO). Usa o MESMO dataset (dataset/final/, formato YOLO com negativas) — nao
precisa converter para COCO-json nem mexer nas pastas.

--- Por que o RetinaNet nao vem do Ultralytics? ---
O Ultralytics so tem YOLO e RT-DETR. RetinaNet vem do proprio torchvision
(torchvision.models.detection), entao este script tem um loop de treino manual
(dataset, dataloader, forward/backward) em vez de um `model.train(...)` pronto.

--- Ponto tecnico importante (diferente do YOLO!) ---
No torchvision, o RetinaNet conta a classe de FUNDO dentro de `num_classes`.
Como so temos 1 classe real (arara_azul), usamos num_classes=2:
    0 = fundo (background, implicito)
    1 = arara_azul (nosso label, YOLO usa 0 -> aqui vira 1)
As imagens negativas (sem .txt) continuam funcionando como fundo: viram um
alvo com boxes/labels vazios (tensor de shape (0,4) e (0,)).

--- Sobre o tamanho das imagens ---
Para manter a comparacao justa com o YOLO (imgsz=640) e nao estourar a
memoria da GPU, cada imagem e redimensionada para IMGSZ x IMGSZ (esticando,
sem letterbox) antes de entrar no modelo. As caixas, que ja estao salvas em
formato normalizado (0-1) no .txt do YOLO, sao convertidas direto pros pixels
do tamanho final -> nao precisa reajustar nada.

--- Sobre as metricas ---
Calculamos mAP com `torchmetrics` e tambem Precision, Recall e F1 com
casamento de caixas por IoU=0.50 e score minimo=0.50. A matriz de confusao
usa TN/FP/FN/TP; TN e contado em nivel de imagem (imagem sem arara real e
sem deteccao). O CSV registra uma linha por epoca. A validacao continua a
cada 5 epocas e na ultima para evitar deixar o treino muito mais lento.

    pip install torch torchvision torchmetrics pycocotools
    python 10_treinar_retinanet.py
(Baixa sozinho os pesos pre-treinados no COCO na 1a vez.)

Saidas:
  runs_retinanet/<NOME>/best.pt                  -> pesos do melhor epoch
  runs_retinanet/<NOME>/last.pt                  -> pesos do ultimo epoch
  runs_retinanet/<NOME>/metricas_por_epoca.csv   -> metricas de cada epoca
  runs_retinanet/<NOME>/matriz_confusao.csv      -> matriz da ultima validacao
  runs_retinanet/<NOME>/matriz_confusao_final.csv -> matriz da avaliacao final
  runs_retinanet/<NOME>/predicoes_validacao/     -> imagens com deteccoes desenhadas
"""

from __future__ import annotations
import time
import csv
from functools import partial
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms import v2 as T

# ------------------------------------------------------------------
DATA_ROOT = Path("dataset/final")   # MESMO dataset do YOLO (com negativas)
IMGSZ = 640
EPOCAS = 60                          # igual ao 4_treinar.py, p/ comparacao justa
BATCH = 4                            # RetinaNet-R50 e mais pesado; RTX 3050 4GB -> comece com 4, se faltar memoria use 2
LR = 0.0005
NOME = "arara_retinanet_r50"
SAIDA = Path("runs_retinanet") / NOME
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NUM_CLASSES = 2   # 0 = fundo (implicito) | 1 = arara_azul

# Visualizacao das deteccoes ao final do treinamento
GERAR_PREDICOES_VISUAIS = True
QTD_IMAGENS_VISUAIS = 10
CONF_THRESHOLD = 0.50
# ------------------------------------------------------------------


# ==================== DATASET (le o mesmo formato do YOLO) ====================
class DatasetYoloParaRetinaNet(Dataset):
    """Le dataset/final/images/<split> e dataset/final/labels/<split>.
    Imagem sem .txt correspondente = negativa -> alvo com boxes/labels vazios."""

    def __init__(self, raiz: Path, split: str, imgsz: int):
        self.dir_img = raiz / "images" / split
        self.dir_lbl = raiz / "labels" / split
        self.imgsz = imgsz
        self.imagens = sorted(p for p in self.dir_img.iterdir() if p.suffix.lower() in IMG_EXT)
        self.transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, idx):
        caminho = self.imagens[idx]
        img = Image.open(caminho).convert("RGB").resize((self.imgsz, self.imgsz))

        lbl_path = self.dir_lbl / f"{caminho.stem}.txt"
        boxes, labels = [], []
        if lbl_path.exists():
            for linha in lbl_path.read_text().strip().splitlines():
                if not linha.strip():
                    continue
                _, xc, yc, bw, bh = (float(v) for v in linha.split())
                # normalizado (0-1) -> pixels no tamanho final (imgsz x imgsz)
                x1 = (xc - bw / 2) * self.imgsz
                y1 = (yc - bh / 2) * self.imgsz
                x2 = (xc + bw / 2) * self.imgsz
                y2 = (yc + bh / 2) * self.imgsz
                boxes.append([x1, y1, x2, y2])
                labels.append(1)   # unica classe real = 1 (0 e fundo)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        alvo = {"boxes": boxes_t, "labels": labels_t}
        return self.transform(img), alvo


def collate_fn(batch):
    imagens, alvos = zip(*batch)
    return list(imagens), list(alvos)


# ==================== MODELO ====================
def criar_modelo(num_classes: int, imgsz: int):
    modelo = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=imgsz, max_size=imgsz,   # nossas imagens ja vem em imgsz x imgsz
    )
    num_anchors = modelo.head.classification_head.num_anchors
    modelo.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32),
    )
    return modelo


# ==================== VALIDACAO E METRICAS ====================
def calcular_iou(box_a, box_b):
    """Calcula IoU entre duas caixas [x1, y1, x2, y2]."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1])
    )
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def calcular_precision_recall_f1(preds, alvos, score_threshold=0.50, iou_threshold=0.50):
    """
    Calcula Precision, Recall e F1 para a classe arara_azul.

    Regra:
      - score >= score_threshold entra na avaliacao;
      - uma predicao casa com no maximo uma caixa real;
      - IoU >= iou_threshold = TP;
      - predicao sem casamento = FP;
      - caixa real sem casamento = FN.

    A matriz de confusao considera:
      TN = imagens sem arara real e sem deteccao;
      FP = deteccoes em imagens sem casamento;
      FN = caixas reais nao detectadas;
      TP = caixas reais corretamente detectadas.
    """
    tp = fp = fn = tn = 0

    for pred, alvo in zip(preds, alvos):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes = alvo["boxes"]

        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]

        matched_gt = set()
        image_tp = 0
        image_fp = 0

        # Ordena implicitamente pela ordem de score, pois o torchvision
        # normalmente retorna as deteccoes em ordem decrescente.
        for pbox in pred_boxes:
            melhor_iou = 0.0
            melhor_gt = -1

            for j, gtbox in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = calcular_iou(pbox, gtbox)
                if iou > melhor_iou:
                    melhor_iou = iou
                    melhor_gt = j

            if melhor_iou >= iou_threshold and melhor_gt >= 0:
                matched_gt.add(melhor_gt)
                tp += 1
                image_tp += 1
            else:
                fp += 1
                image_fp += 1

        faltantes = len(gt_boxes) - len(matched_gt)
        fn += faltantes

        # Para TN/FP da matriz de confusao em nivel de imagem:
        # uma imagem sem GT e sem predicao e um verdadeiro negativo.
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@torch.no_grad()
def avaliar(modelo, loader, device, score_threshold=0.50, iou_threshold=0.50):
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        print("torchmetrics nao instalado (pip install torchmetrics pycocotools) -> pulando mAP.")
        return None

    metrica = MeanAveragePrecision(iou_type="bbox")
    modelo.eval()

    todos_preds = []
    todos_alvos = []

    for imagens, alvos in loader:
        imagens = [img.to(device) for img in imagens]
        preds = modelo(imagens)

        preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
        alvos_cpu = [{k: v.cpu() for k, v in a.items()} for a in alvos]

        metrica.update(preds_cpu, alvos_cpu)
        todos_preds.extend(preds_cpu)
        todos_alvos.extend(alvos_cpu)

    metricas_map = metrica.compute()
    metricas_pr = calcular_precision_recall_f1(
        todos_preds,
        todos_alvos,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )

    resultado = {
        "map50": float(metricas_map["map_50"]),
        "map50_95": float(metricas_map["map"]),
        "mar_100": float(metricas_map["mar_100"]),
        **metricas_pr,
    }

    return resultado


def salvar_matriz_confusao(metricas, caminho):
    """
    Salva a matriz de confusao como CSV:
                 Predito negativo | Predito positivo
    Real negativo       TN        |       FP
    Real positivo       FN        |       TP
    """
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "Predito negativo", "Predito positivo"])
        writer.writerow(["Real negativo", metricas["tn"], metricas["fp"]])
        writer.writerow(["Real positivo", metricas["fn"], metricas["tp"]])


# ==================== PREDICAO / VISUALIZACAO ====================
@torch.no_grad()
def prever_e_salvar_imagem(modelo, caminho_imagem, caminho_saida, device,
                           conf_threshold=0.50):
    """
    Executa o RetinaNet em uma imagem e salva a imagem com as deteccoes.

    A imagem e redimensionada para IMGSZ x IMGSZ, da mesma forma que no
    treinamento. A caixa retornada pelo modelo e desenhada na imagem.
    """
    modelo.eval()

    imagem_original = Image.open(caminho_imagem).convert("RGB")
    imagem_modelo = imagem_original.resize((IMGSZ, IMGSZ))

    tensor = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])(imagem_modelo)

    pred = modelo([tensor.to(device)])[0]

    boxes = pred["boxes"].detach().cpu()
    scores = pred["scores"].detach().cpu()
    labels = pred["labels"].detach().cpu()

    # Desenha na imagem 640x640 para manter a mesma escala usada pelo modelo.
    imagem_vis = imagem_modelo.copy()
    draw = ImageDraw.Draw(imagem_vis)

    deteccoes = 0

    for box, score, label in zip(boxes, scores, labels):
        score = float(score)

        if score < conf_threshold:
            continue

        # RetinaNet: label 1 = arara_azul.
        if int(label) != 1:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.tolist()]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        texto = f"arara_azul {score:.2f}"

        # Caixa de fundo para o texto.
        try:
            fonte = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            fonte = ImageFont.load_default()

        bbox_texto = draw.textbbox((x1, max(0, y1 - 22)), texto, font=fonte)
        draw.rectangle(bbox_texto, fill="red")
        draw.text(
            (x1, max(0, y1 - 22)),
            texto,
            fill="white",
            font=fonte
        )

        deteccoes += 1

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    imagem_vis.save(caminho_saida)

    print(
        f"  -> {Path(caminho_imagem).name}: "
        f"{deteccoes} deteccao(oes) | salvo em {caminho_saida}"
    )


def gerar_imagens_validacao(modelo, dataset_val, device,
                            quantidade=10, conf_threshold=0.50):
    """
    Gera visualizacoes para algumas imagens da validacao.

    Por padrao, usa as primeiras 'quantidade' imagens do conjunto de validacao.
    """
    pasta_pred = SAIDA / "predicoes_validacao"
    pasta_pred.mkdir(parents=True, exist_ok=True)

    quantidade = min(quantidade, len(dataset_val))

    print("\n==== GERANDO PREDICOES VISUAIS ====")
    print(f"Imagens: {quantidade} | confidence threshold: {conf_threshold}")

    for i in range(quantidade):
        caminho = dataset_val.imagens[i]
        saida = pasta_pred / f"{caminho.stem}_retinanet.jpg"

        prever_e_salvar_imagem(
            modelo=modelo,
            caminho_imagem=caminho,
            caminho_saida=saida,
            device=device,
            conf_threshold=conf_threshold,
        )

# ==================== TREINO ====================
def main():
    data_yaml = DATA_ROOT / "data.yaml"
    if not data_yaml.exists():
        print(f"Nao achei {data_yaml}. Rode o 7_preparar_com_negativas.py (ou 3_preparar_dataset.py) antes.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Treinando na GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Sem GPU detectada -> CPU (RetinaNet fica bem lento assim).")

    treino = DatasetYoloParaRetinaNet(DATA_ROOT, "train", IMGSZ)
    val = DatasetYoloParaRetinaNet(DATA_ROOT, "val", IMGSZ)
    print(f"Treino: {len(treino)} imagens | Validacao: {len(val)} imagens")

    loader_treino = DataLoader(treino, batch_size=BATCH, shuffle=True,
                                collate_fn=collate_fn, num_workers=2)
    loader_val = DataLoader(val, batch_size=BATCH, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)

    modelo = criar_modelo(NUM_CLASSES, IMGSZ).to(device)
    params = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=EPOCAS)

    SAIDA.mkdir(parents=True, exist_ok=True)

    # CSV com uma linha por epoca.
    caminho_csv = SAIDA / "metricas_por_epoca.csv"
    campos_csv = [
        "epoca", "loss", "lr", "tempo_s",
        "precision", "recall", "f1",
        "mAP50", "mAP50-95", "mar_100",
        "TP", "FP", "FN", "TN"
    ]

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(campos_csv)

    melhor_map50 = -1.0

    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        t0 = time.time()
        perda_acum = 0.0

        for imagens, alvos in loader_treino:
            imagens = [img.to(device) for img in imagens]
            alvos = [{k: v.to(device) for k, v in a.items()} for a in alvos]

            perdas = modelo(imagens, alvos)
            perda_total = sum(perdas.values())

            otimizador.zero_grad()
            perda_total.backward()
            otimizador.step()
            perda_acum += perda_total.item()

        scheduler.step()
        perda_media = perda_acum / max(1, len(loader_treino))
        tempo_epoca = time.time() - t0
        lr_atual = scheduler.get_last_lr()[0]

        print(f"[epoca {epoca}/{EPOCAS}] loss={perda_media:.4f} "
              f"({tempo_epoca:.1f}s) lr={lr_atual:.6f}")

        # A validacao continua a cada 5 epocas e na ultima, para nao deixar
        # o treinamento muito mais lento.
        metricas = None
        if epoca % 5 == 0 or epoca == EPOCAS:
            metricas = avaliar(
                modelo,
                loader_val,
                device,
                score_threshold=0.50,
                iou_threshold=0.50,
            )

            if metricas is not None:
                print(
                    f"  -> validacao: "
                    f"Precision={metricas['precision']:.3f} "
                    f"Recall={metricas['recall']:.3f} "
                    f"F1={metricas['f1']:.3f} | "
                    f"mAP50={metricas['map50']:.3f} "
                    f"mAP50-95={metricas['map50_95']:.3f} "
                    f"mar_100={metricas['mar_100']:.3f}"
                )
                print(
                    f"     matriz: TN={metricas['tn']} "
                    f"FP={metricas['fp']} "
                    f"FN={metricas['fn']} "
                    f"TP={metricas['tp']}"
                )

                if metricas["map50"] > melhor_map50:
                    melhor_map50 = metricas["map50"]
                    torch.save(modelo.state_dict(), SAIDA / "best.pt")
                    print(f"  -> novo melhor modelo salvo ({SAIDA / 'best.pt'})")

                # A matriz abaixo representa a validacao mais recente.
                salvar_matriz_confusao(
                    metricas,
                    SAIDA / "matriz_confusao.csv"
                )

        # Cada epoca entra no CSV. Nas epocas sem validacao, as colunas de
        # metricas ficam vazias.
        linha = {
            "epoca": epoca,
            "loss": f"{perda_media:.6f}",
            "lr": f"{lr_atual:.8f}",
            "tempo_s": f"{tempo_epoca:.2f}",
            "precision": "" if metricas is None else f"{metricas['precision']:.6f}",
            "recall": "" if metricas is None else f"{metricas['recall']:.6f}",
            "f1": "" if metricas is None else f"{metricas['f1']:.6f}",
            "mAP50": "" if metricas is None else f"{metricas['map50']:.6f}",
            "mAP50-95": "" if metricas is None else f"{metricas['map50_95']:.6f}",
            "mar_100": "" if metricas is None else f"{metricas['mar_100']:.6f}",
            "TP": "" if metricas is None else metricas["tp"],
            "FP": "" if metricas is None else metricas["fp"],
            "FN": "" if metricas is None else metricas["fn"],
            "TN": "" if metricas is None else metricas["tn"],
        }

        with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=campos_csv).writerow(linha)

        torch.save(modelo.state_dict(), SAIDA / "last.pt")

    print("\n==== TREINO CONCLUIDO ====")

    # Avaliacao final completa.
    metricas_finais = avaliar(
        modelo,
        loader_val,
        device,
        score_threshold=0.50,
        iou_threshold=0.50,
    )

    if metricas_finais is not None:
        print("\n==== METRICAS FINAIS — RetinaNet-R50 (validacao, ultimo epoch) ====")
        print(f"Precision:         {metricas_finais['precision']:.3f}")
        print(f"Recall:            {metricas_finais['recall']:.3f}")
        print(f"F1-score:          {metricas_finais['f1']:.3f}")
        print(f"mAP50:             {metricas_finais['map50']:.3f}")
        print(f"mAP50-95:          {metricas_finais['map50_95']:.3f}")
        print(f"mar_100 (recall):  {metricas_finais['mar_100']:.3f}")
        print(
            f"Matriz: TN={metricas_finais['tn']} | "
            f"FP={metricas_finais['fp']} | "
            f"FN={metricas_finais['fn']} | "
            f"TP={metricas_finais['tp']}"
        )

        salvar_matriz_confusao(
            metricas_finais,
            SAIDA / "matriz_confusao_final.csv"
        )

    # Gera imagens com bounding boxes usando o modelo do ultimo epoch.
    if GERAR_PREDICOES_VISUAIS:
        gerar_imagens_validacao(
            modelo=modelo,
            dataset_val=val,
            device=device,
            quantidade=QTD_IMAGENS_VISUAIS,
            conf_threshold=CONF_THRESHOLD,
        )

    print(f"\nCSV de metricas: {caminho_csv}")
    print(f"Matriz de confusao: {SAIDA / 'matriz_confusao_final.csv'}")
    print(f"Pesos em: {SAIDA}/best.pt (melhor) e {SAIDA}/last.pt (ultimo)")


if __name__ == "__main__":
    main()
