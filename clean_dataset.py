import os, cv2
from PIL import Image
import imagehash

DATASET_PATH = "MySkinData"
CLEAN_PATH = "CleanData"

os.makedirs(CLEAN_PATH, exist_ok=True)

MIN_SIZE = 100
BLUR_THRESHOLD = 50
HASHES = set()

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

for cls in os.listdir(DATASET_PATH):
    inp = os.path.join(DATASET_PATH, cls)
    out = os.path.join(CLEAN_PATH, cls)
    os.makedirs(out, exist_ok=True)

    for file in os.listdir(inp):
        path = os.path.join(inp, file)

        try:
            img = cv2.imread(path)
            if img is None:
                continue

            h, w = img.shape[:2]
            if h < MIN_SIZE or w < MIN_SIZE:
                continue

            if is_blurry(img):
                continue

            pil = Image.open(path).convert("RGB")
            hsh = imagehash.average_hash(pil)

            if hsh in HASHES:
                continue

            HASHES.add(hsh)
            pil.save(os.path.join(out, file))

        except:
            pass

print("✅ Clean dataset ready")