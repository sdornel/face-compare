import cv2
import numpy as np
import os
import pandas as pd
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# -------- config --------
IMAGE_DIR = "faces"
MODEL_NAME = "buffalo_l"
CSV_OUTPUT = "similarity_matrix.csv"
# ------------------------

print("Loading face model...")
app = FaceAnalysis(name=MODEL_NAME, providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

def display_stability_warnings(similarity, image_names):
    THRESHOLD = 0.60
    WEAK_RATIO_LIMIT = 0.4
    print("\n=== Identity Stability Warnings ===\n")

    for i, name in enumerate(image_names):
        sims = similarity[i]

        # exclude self-comparison
        others = np.delete(sims, i)
        other_names = [n for j, n in enumerate(image_names) if j != i]

        avg_sim = others.mean()

        weak_pairs = [
            (other_names[j], others[j])
            for j in range(len(others))
            if others[j] < THRESHOLD
        ]

        weak_ratio = len(weak_pairs) / len(others)

        if avg_sim < THRESHOLD or weak_ratio > WEAK_RATIO_LIMIT:
            print(f"⚠️  {name}")
            print(f"    avg similarity: {avg_sim:.3f}")
            print(f"    weak pairs (<{THRESHOLD}): {len(weak_pairs)}/{len(others)}")

            for other_name, score in weak_pairs:
                print(f"       ↳ {other_name}: {score:.3f}")

            print()

def main():
    embeddings = []
    image_names = []
    print("\nProcessing images:\n")

    for file in sorted(os.listdir(IMAGE_DIR)):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(IMAGE_DIR, file)
        img = cv2.imread(path)

        faces = app.get(img)

        if len(faces) == 0:
            print(f"❌ No face detected: {file}")
            continue

        # choose largest detected face
        face = sorted(faces, key=lambda f: f.bbox[2]*f.bbox[3], reverse=True)[0]

        embeddings.append(face.embedding)
        image_names.append(file)

        print(f"✅ {file}")

    embeddings = np.array(embeddings)

    print("\nComputing similarity matrix...\n")
    similarity = cosine_similarity(embeddings)

    display_stability_warnings(similarity, image_names)

    # ---------- print table ----------
    header = " " * 22
    for name in image_names:
        header += f"{name[:10]:>12}"
    print(header)

    for i, row in enumerate(similarity):
        line = f"{image_names[i]:<22}"
        for val in row:
            line += f"{val:12.3f}"
        print(line)

    # ---------- save CSV ----------
    df = pd.DataFrame(similarity, index=image_names, columns=image_names)
    df.to_csv(CSV_OUTPUT)

    print(f"\n📄 Saved CSV to: {CSV_OUTPUT}")

main()