"""Run a reproducible YOLOv7 training and final evaluation on audited D-Fire data."""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def run(command, cwd, logfile):
    with logfile.open("w", encoding="utf-8") as output:
        output.write("COMMAND: " + " ".join(map(str, command)) + "\n")
        output.flush()
        process = subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(f"Comando falhou ({process.returncode}); veja {logfile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolov7", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="dfire.yaml gerado pela auditoria")
    parser.add_argument("--weights", type=Path, required=True, help="yolov7_training.pt oficial")
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--freeze-layers", type=int, default=50)
    parser.add_argument("--run-name", default="dfire-baseline")
    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    compatibility = Path(__file__).resolve().parent / "yolov7_compat"
    os.environ["PYTHONPATH"] = str(compatibility) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if min(args.epochs, args.batch_size, args.img_size) < 1 or args.freeze_layers < 0:
        parser.error("epochs, batch-size e img-size devem ser positivos; freeze-layers não pode ser negativo")
    if not args.run_name or any(char in args.run_name for char in "/\\"):
        parser.error("--run-name deve ser um nome simples de pasta")
    yolov7, data, weights, output = (p.resolve() for p in (args.yolov7, args.data, args.weights, args.output))
    for required in (yolov7 / "train.py", yolov7 / "test.py", data, weights):
        if not required.is_file():
            parser.error(f"Arquivo ausente: {required}")
    test_name = args.run_name + "-test"
    if (output / args.run_name).exists() or (output / test_name).exists():
        parser.error("A pasta de execução já existe; use outro --output para preservar os resultados")
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=yolov7, capture_output=True, text=True, check=True).stdout.strip()
    metadata = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "yolov7_commit": commit, "python": sys.version, "platform": platform.platform(),
                "data": str(data), "weights": str(weights), "epochs": args.epochs,
                "batch_size": args.batch_size, "img_size": args.img_size, "workers": args.workers,
                "device": args.device, "freeze_layers": args.freeze_layers,
                "run_name": args.run_name,
                "note": "YOLOv7 train.py usa init_seeds(2 + rank); com rank=-1, seed=1. Teste executado uma vez após o treino."}
    try:
        import torch
        metadata["torch"] = torch.__version__
        metadata["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        metadata["torch"] = None
    (output / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    train = [sys.executable, "train.py", "--workers", str(args.workers), "--device", args.device,
             "--batch-size", str(args.batch_size), "--data", str(data), "--img", str(args.img_size), str(args.img_size),
             "--cfg", "cfg/training/yolov7.yaml", "--weights", str(weights),
             "--name", args.run_name, "--project", str(output),
             "--hyp", "data/hyp.scratch.custom.yaml", "--epochs", str(args.epochs),
             "--freeze", str(args.freeze_layers)]
    run(train, yolov7, output / "train.log")
    best = output / args.run_name / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Checkpoint não encontrado: {best}")
    test = [sys.executable, "test.py", "--data", str(data), "--img-size", str(args.img_size),
            "--batch-size", str(args.batch_size), "--device", args.device, "--weights", str(best),
            "--task", "test", "--name", test_name, "--project", str(output), "--verbose"]
    run(test, yolov7, output / "test.log")
    print(f"Execução concluída. Logs e métricas em {output}")


if __name__ == "__main__":
    main()
