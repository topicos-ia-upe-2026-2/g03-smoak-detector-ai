"""Audit an official D-Fire split and generate a YOLOv7 data file.

No source image or label is modified. The audit fails closed on malformed data.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
BOUNDARY_EPSILON = 1e-9  # tolerância apenas para arredondamento decimal após recorte


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(root, class_names):
    records = []
    errors = []
    hashes = defaultdict(list)
    for split in SPLITS:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            errors.append(f"{split}: esperado images/ e labels/ em {root / split}")
            continue
        images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in SUFFIXES)
        if not images:
            errors.append(f"{split}: nenhuma imagem encontrada")
        seen_labels = set()
        for image in images:
            relative = image.relative_to(images_dir)
            label = labels_dir / relative.with_suffix(".txt")
            key = f"{split}/{relative.as_posix()}"
            classes = set()
            boxes = Counter()
            try:
                with Image.open(image) as opened:
                    opened.verify()
                with Image.open(image) as opened:
                    width, height = opened.size
                if width < 1 or height < 1:
                    raise ValueError("dimensões inválidas")
                image_hash = sha256(image)
                hashes[image_hash].append((split, key))
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                errors.append(f"{key}: imagem inválida: {exc}")
                continue
            if not label.is_file():
                errors.append(f"{key}: rótulo ausente (use .txt vazio para imagem negativa)")
                continue
            seen_labels.add(label.resolve())
            for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("cada caixa exige classe x y w h")
                    cls = int(parts[0])
                    if str(cls) != parts[0] or cls not in (0, 1):
                        raise ValueError("classe deve ser 0 ou 1")
                    x, y, w, h = map(float, parts[1:])
                    if not (0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
                        raise ValueError("coordenadas fora do intervalo YOLO")
                    if not (-BOUNDARY_EPSILON <= x - w / 2 and x + w / 2 <= 1 + BOUNDARY_EPSILON
                            and -BOUNDARY_EPSILON <= y - h / 2 and y + h / 2 <= 1 + BOUNDARY_EPSILON):
                        raise ValueError("caixa ultrapassa a imagem")
                    classes.add(cls)
                    boxes[cls] += 1
                except ValueError as exc:
                    errors.append(f"{key}:{line_number}: {exc}")
            category = "ambos" if len(classes) == 2 else class_names[next(iter(classes))] if classes else "nenhum"
            records.append({"split": split, "image": key, "sha256": image_hash,
                            "width": width, "height": height, "category": category,
                            "boxes_0": boxes[0], "boxes_1": boxes[1]})
        for label in labels_dir.rglob("*.txt"):
            if label.resolve() not in seen_labels:
                errors.append(f"{split}: rótulo sem imagem correspondente: {label}")
    for digest, occurrences in hashes.items():
        if len(occurrences) > 1:
            errors.append(f"duplicata SHA-256 {digest}: " + ", ".join(key for _, key in occurrences))
    return records, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Pasta com train/val/test, cada qual com images/labels")
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_audit"))
    parser.add_argument("--class-zero", choices=("fumaça", "fogo"), required=True)
    parser.add_argument("--class-one", choices=("fumaça", "fogo"), required=True)
    args = parser.parse_args()
    if args.class_zero == args.class_one:
        parser.error("As classes 0 e 1 devem ser diferentes")
    root = args.root.resolve()
    records, errors = inspect(root, (args.class_zero, args.class_one))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("split", "image", "sha256", "width", "height", "category", "boxes_0", "boxes_1"))
        writer.writeheader()
        writer.writerows(records)
    summary = {"source": str(root), "class_names": [args.class_zero, args.class_one],
               "counts": {split: dict(Counter(row["category"] for row in records if row["split"] == split)) for split in SPLITS},
               "boxes": {split: {str(cls): sum(row[f"boxes_{cls}"] for row in records if row["split"] == split) for cls in (0, 1)} for split in SPLITS},
               "images": len(records), "errors": errors}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        print(f"Auditoria falhou com {len(errors)} erro(s). Veja {output / 'summary.json'}")
        raise SystemExit(1)
    yaml = "\n".join([f"train: {json.dumps(str(root / 'train' / 'images'), ensure_ascii=False)}",
                      f"val: {json.dumps(str(root / 'val' / 'images'), ensure_ascii=False)}",
                      f"test: {json.dumps(str(root / 'test' / 'images'), ensure_ascii=False)}", "nc: 2",
                      f"names: {json.dumps([args.class_zero, args.class_one], ensure_ascii=False)}", ""])
    (output / "dfire.yaml").write_text(yaml, encoding="utf-8")
    print(f"Auditoria aprovada: {len(records)} imagens. YAML: {output / 'dfire.yaml'}")


if __name__ == "__main__":
    main()
