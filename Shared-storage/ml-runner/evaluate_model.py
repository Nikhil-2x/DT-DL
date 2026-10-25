"""Check a trained checkpoint (.pt) on your own machine.

    python evaluate_model.py --script examples/image_classifier.py \\
        --checkpoint model.pt --data test_images.zip [--image one.png]

--script       the SAME project script you trained with (it rebuilds the model)
--checkpoint   the .pt file downloaded from the dashboard (Model button)
--data         data to test on: a .zip or a folder, laid out like the training
               data. Use images the model has NOT seen, otherwise the score
               only shows how well it memorised the training set.
--image        optional: classify a single image and show the top guesses

It prints the accuracy of the trained model next to an untrained copy of the
same network (the baseline), per-class accuracy and a confusion matrix.
Checkpoints are pickles: only load files from jobs you trust.
"""
import argparse
import os
import sys
import tempfile
import zipfile

import torch
from torch.utils.data import DataLoader

import project_loader as pl


def prepare_data_dir(path: str) -> str:
    if os.path.isdir(path):
        return path
    if path.lower().endswith(".zip"):
        out = tempfile.mkdtemp(prefix="eval_data_")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out)
        return out
    raise SystemExit(f"--data must be a folder or a .zip file, got {path}")


def predict_all(model, dataset, batch_size=128):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=batch_size):
            preds.append(model(x).argmax(1))
            targets.append(torch.as_tensor(y))
    return torch.cat(preds), torch.cat(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    data_dir = prepare_data_dir(args.data) if args.data else None
    module = pl.load_module(args.script)
    pl.check_functions(module)
    dataset = pl.make_dataset(module, data_dir)
    model = pl.make_model(module, dataset)

    # Untrained copy: the baseline the trained model must beat.
    torch.manual_seed(0)
    baseline = pl.make_model(module, dataset)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])  # fails loudly if the architecture differs
    print(f"checkpoint: epoch {ck['epoch']}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters, "
          f"tested on {len(dataset)} samples")

    classes = list(getattr(dataset, "classes", [])) or None
    preds, targets = predict_all(model, dataset)
    base_preds, _ = predict_all(baseline, dataset)
    acc = (preds == targets).float().mean().item()
    base_acc = (base_preds == targets).float().mean().item()
    print(f"\naccuracy: trained {acc:.1%}   vs   untrained baseline {base_acc:.1%}")

    n = int(max(targets.max(), preds.max())) + 1
    names = classes if classes and len(classes) == n else [str(i) for i in range(n)]
    confusion = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(targets, preds):
        confusion[t, p] += 1
    print("\nper-class accuracy:")
    for i, name in enumerate(names):
        total = int(confusion[i].sum())
        if total:
            print(f"  {name:<14} {confusion[i, i] / total:.1%}  ({int(confusion[i, i])}/{total})")
    print("\nconfusion matrix (rows = true class, columns = predicted):")
    print("  " + " " * 14 + "".join(f"{nm[:8]:>9}" for nm in names))
    for i, name in enumerate(names):
        print(f"  {name:<14}" + "".join(f"{int(v):>9}" for v in confusion[i]))

    if args.image:
        from PIL import Image

        tfm = getattr(dataset, "transform", None)
        if tfm is None:
            raise SystemExit("--image needs a dataset with a .transform (e.g. ImageFolder)")
        x = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x), 1)[0]
        print(f"\n{args.image}:")
        for p, i in zip(*probs.topk(min(3, n))):
            print(f"  {names[int(i)]:<14} {p:.1%}")


if __name__ == "__main__":
    try:
        main()
    except pl.ContractError as e:
        sys.exit(f"PROJECT ERROR: {e}")
