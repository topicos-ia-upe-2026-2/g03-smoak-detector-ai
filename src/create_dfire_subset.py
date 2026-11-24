"""Create a deterministic, category-stratified D-Fire training subset."""

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png"}


def category(label):
    classes = {line.split()[0] for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    if classes == {"0", "1"}:
        return "ambos"
    if classes == {"0"}:
        return "fumaça"
    if classes == {"1"}:
        return "fogo"
    if not classes:
        return "nenhum"
    raise ValueError(f"classes inesperadas em {label}: {sorted(classes)}")


def allocate(counts, total):
    available = sum(counts.values())
    exact = {key: total * value / available for key, value in counts.items()}
    result = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(counts, key=lambda key: (exact[key] - result[key], counts[key], key), reverse=True)
    for key in order[:remaining]:
        result[key] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dados limpos com train/val/test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.size < 1:
        parser.error("--size deve ser positivo")
    if output.exists() and any(output.iterdir()):
        parser.error("--output deve estar vazio")
    images_dir, labels_dir = root / "train" / "images", root / "train" / "labels"
    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in EXTENSIONS)
    if args.size > len(images):
        parser.error(f"--size excede as {len(images)} imagens de treino")
    groups = {}
    for image in images:
        relative = image.relative_to(images_dir)
        label = labels_dir / relative.with_suffix(".txt")
        if not label.is_file():
            parser.error(f"rótulo ausente: {label}")
        groups.setdefault(category(label), []).append((image, label, relative))
    quotas = allocate({key: len(value) for key, value in groups.items()}, args.size)
    rng = random.Random(args.seed)
    selected = []
    for key in sorted(groups):
        candidates = groups[key][:]
        rng.shuffle(candidates)
        selected.extend((key, *item) for item in candidates[:quotas[key]])
    selected.sort(key=lambda item: item[3].as_posix())
    rows = []
    for key, image, label, relative in selected:
        target_image = output / "train" / "images" / relative
        target_label = output / "train" / "labels" / relative.with_suffix(".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(image.resolve(), target_image)
        except (OSError, NotImplementedError):
            try:
                os.link(image, target_image)
            except OSError:
                shutil.copy2(image, target_image)
        shutil.copy2(label, target_label)
        rows.append({"image": relative.as_posix(), "category": key})
    with (output / "selection.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("image", "category"))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"source": str(root), "seed": args.seed, "size": args.size,
               "source_counts": {key: len(value) for key, value in groups.items()},
               "selected_counts": dict(Counter(row["category"] for row in rows))}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    yaml = "\n".join([f'train: "{output / "train" / "images"}"',
                      f'val: "{root / "val" / "images"}"',
                      f'test: "{root / "test" / "images"}"',
                      "nc: 2", "names: ['fumaça', 'fogo']", ""])
    (output / "dfire_subset.yaml").write_text(yaml, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
