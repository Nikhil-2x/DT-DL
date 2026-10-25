"""Generate a small image dataset to try examples/image_classifier.py.

    python examples/make_shapes_dataset.py          # writes shapes.zip

Three classes (circle, square, triangle) of 48x48 colour images with random
position, size, colour and noise: easy enough to learn in a few epochs, and
laid out exactly like a real dataset (one folder per class).
"""
import io
import random
import sys
import zipfile

from PIL import Image, ImageDraw

CLASSES = ["circle", "square", "triangle"]
SIZE = 48


def draw(cls, rng):
    img = Image.new("RGB", (SIZE, SIZE), tuple(rng.randint(0, 80) for _ in range(3)))
    d = ImageDraw.Draw(img)
    r = rng.randint(9, 17)
    cx, cy = rng.randint(r, SIZE - r), rng.randint(r, SIZE - r)
    fill = tuple(rng.randint(140, 255) for _ in range(3))
    if cls == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif cls == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
    px = img.load()
    for _ in range(60):  # speckle noise
        px[rng.randrange(SIZE), rng.randrange(SIZE)] = tuple(rng.randint(0, 255) for _ in range(3))
    return img


def main(path="shapes.zip", per_class=200, seed=3):
    rng = random.Random(seed)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for cls in CLASSES:
            for i in range(per_class):
                buf = io.BytesIO()
                draw(cls, rng).save(buf, "PNG")
                zf.writestr(f"shapes/{cls}/{i:03d}.png", buf.getvalue())
    print(f"wrote {path}: {len(CLASSES)} classes x {per_class} images")


if __name__ == "__main__":
    main(*sys.argv[1:2])
