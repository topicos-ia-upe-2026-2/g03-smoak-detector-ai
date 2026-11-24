"""Create a YOLO-compatible copy of D-Fire labels without changing source files."""

import argparse
import csv
import math
import os
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def clean_line(line):
    parts = line.split()
    if len(parts) != 5 or parts[0] not in ("0", "1"):
        raise ValueError("esperado: classe(0/1) x y largura altura")
    x, y, w, h = map(float, parts[1:])
    if not all(map(math.isfinite, (x, y, w, h))):
        raise ValueError("coordenadas não finitas")
    if w <= 0 or h <= 0:
        return None, "caixa com área zero ou negativa"
    x1, x2 = max(0.0, x - w / 2), min(1.0, x + w / 2)
    y1, y2 = max(0.0, y - h / 2), min(1.0, y + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None, "caixa fora da imagem"
    area_fraction = ((x2 - x1) * (y2 - y1)) / (w * h)
    if area_fraction < 0.8:
        raise ValueError(f"recorte preserva apenas {area_fraction:.3f} da área; revisar manualmente")
    clipped = (x1 != x - w / 2 or x2 != x + w / 2 or y1 != y - h / 2 or y2 != y + h / 2)
    if clipped:
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    result = f"{parts[0]} {x:.10f} {y:.10f} {w:.10f} {h:.10f}"
    return result, "recortada aos limites" if clipped else "inalterada"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if root == output or root in output.parents or output in root.parents:
        parser.error("--output deve ficar fora da árvore de entrada")
    if output.exists() and any(output.iterdir()):
        parser.error("--output deve estar vazio para preservar execuções anteriores")
    report = []
    counts = {}
    for split in SPLITS:
        images_dir, labels_dir = root / split / "images", root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            parser.error(f"Partição ausente: {split}/images ou {split}/labels")
        images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)
        if not images:
            parser.error(f"Sem imagens em {images_dir}")
        counts[split] = len(images)
        for image in images:
            relative = image.relative_to(images_dir)
            label = labels_dir / relative.with_suffix(".txt")
            if not label.is_file():
                parser.error(f"Rótulo ausente: {label}")
            lines = []
            for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    cleaned, action = clean_line(line)
                except ValueError as exc:
                    parser.error(f"{label}:{number}: {exc}")
                if cleaned is not None:
                    lines.append(cleaned)
                if action != "inalterada":
                    report.append({"image": f"{split}/{relative.as_posix()}", "line": number,
                                   "action": action, "before": line, "after": cleaned or ""})
            target_image = output / split / "images" / relative
            target_label = output / split / "labels" / relative.with_suffix(".txt")
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(image.resolve(), target_image)
            except (OSError, NotImplementedError):
                try:
                    os.link(image, target_image)
                except OSError:
                    shutil.copy2(image, target_image)
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    with (output / "cleaning.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("image", "line", "action", "before", "after"))
        writer.writeheader()
        writer.writerows(report)
    print(f"Preparação concluída: {counts}; {len(report)} correções/remoções. "
          f"Relatório: {output / 'cleaning.csv'}")


if __name__ == "__main__":
    main()
