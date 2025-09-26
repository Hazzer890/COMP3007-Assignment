import os
import re
from ultralytics import YOLO
from PIL import Image

# -------- config --------
INPUT_DIR = "/home/20607591/COMP3007-Assignment/inputImages"     # e.g. "/home/20607591/COMP3007-Assignment/input"
OUTPUT_DIR = "/home/20607591/COMP3007-Assignment/outputImages"   # e.g. "/home/20607591/COMP3007-Assignment/output"
MODEL_PATH = "/home/20607591/COMP3007-Assignment/myModel/building_numbers/weights/best.pt"
# ------------------------

def ensure_dir(d):
    if not os.path.exists(d):
        print(f"Creating directory: {d}")
        os.makedirs(d)

def get_index(filename):
    # matches img<number>.jpg
    m = re.match(r"im(\d+)\.jpg$", filename)
    print(f"Extracted index: {m.group(1)}" if m else "No match found")
    return m.group(1) if m else None

def main():
    ensure_dir(OUTPUT_DIR)
    model = YOLO(MODEL_PATH)
    print("Model loaded.")

    for fileName in sorted(os.listdir(INPUT_DIR)):
        idx = get_index(fileName)
        print(f"Processing file: {fileName}, Index: {idx}")
        if idx is None:
            continue

        img_path = os.path.join(INPUT_DIR, fileName)
        results = model(img_path)

        # We expect exactly one detection
        boxes = results[0].boxes
        if len(boxes) != 1:
            # no output if zero or multiple
            continue

        # load full image
        img = Image.open(img_path).convert("RGB")
        # xyxy gives (x1, y1, x2, y2) as floats
        x1, y1, x2, y2 = boxes.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        crop = img.crop((x1, y1, x2, y2))
        out_fileName = f"bn{idx}.png"
        crop.save(os.path.join(OUTPUT_DIR, out_fileName))

if __name__ == "__main__":
    main()