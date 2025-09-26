import os
import re
from ultralytics import YOLO
from PIL import Image

# -------- config --------
INPUT_DIR = "inputImages"
OUTPUT_DIR = "outputImages"
MODEL_PATH = "myModel/building_numbers/weights/best.pt"
# ------------------------

def ensure_dir(dir):
    if not os.path.exists(dir):
        print(f"Creating directory: {dir}")
        os.makedirs(dir)

def get_index(filename):
    # matches img<number>.jpg
    match = re.match(r"im(\d+)\.jpg$", filename)
    print(f"Extracted index: {match.group(1)}" if match else "No match found")
    return match.group(1) if match else None

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
        # skip if no detections
        if len(boxes) == 0:
            continue
        # if multiple detections, keep only the most confident one
        if len(boxes) > 1:
            confidences = boxes.conf  # Tensor of confidence scores
            max_idx = int(confidences.argmax())
            boxes = boxes[max_idx : max_idx + 1]

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