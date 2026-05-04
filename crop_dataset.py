import os
from PIL import Image

INPUT = "CleanData"
OUTPUT = "CroppedData"

os.makedirs(OUTPUT, exist_ok=True)

def crop(img):
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return img.crop((left, top, left+m, top+m))

for cls in os.listdir(INPUT):
    inp = os.path.join(INPUT, cls)
    out = os.path.join(OUTPUT, cls)
    os.makedirs(out, exist_ok=True)

    for file in os.listdir(inp):
        try:
            img = Image.open(os.path.join(inp, file)).convert("RGB")
            img = crop(img)
            img = img.resize((224,224))
            img.save(os.path.join(out, file))
        except:
            pass

print("✅ Cropped dataset ready")