import cv2
import numpy as np
import os
import pandas as pd
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# -------- config --------
IMAGE_DIR = "images"
MODEL_NAME = "buffalo_l"
CSV_OUTPUT = "similarity_matrix.csv"
# ------------------------

THRESHOLD = 0.55
WEAK_RATIO_LIMIT = 0.4

print("Loading face model...")
app = FaceAnalysis(name=MODEL_NAME, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

def compute_similarity_stats(similarity):
    n = similarity.shape[0]
    values = []

    for i in range(n):
        for j in range(i + 1, n):
            values.append(similarity[i][j])

    values = np.array(values)

    avg = float(np.mean(values))
    median = float(np.median(values))
    minimum = float(np.min(values))
    maximum = float(np.max(values))

    return avg, median, minimum, maximum, values

def display_stability_warnings(similarity, image_names):
    weak_pairs_global = []

    print("\n=== Identity Stability Warnings ===\n")

    for i, name in enumerate(image_names):
        sims = similarity[i]

        others = np.delete(sims, i)
        other_names = [n for j, n in enumerate(image_names) if j != i]

        avg_sim = others.mean()

        weak_pairs = [
            (other_names[j], others[j])
            for j in range(len(others))
            if others[j] < THRESHOLD
        ]

        weak_ratio = len(weak_pairs) / len(others)

        # collect unique weak pairs
        for j in range(i + 1, len(image_names)):
            score = similarity[i][j]
            if score < THRESHOLD:
                weak_pairs_global.append(
                    (image_names[i], image_names[j], score)
                )

        if avg_sim < THRESHOLD or weak_ratio > WEAK_RATIO_LIMIT:
            print(f"⚠️  {name}")
            print(f"    avg similarity: {avg_sim:.3f}")
            print(f"    weak pairs (<{THRESHOLD}): {len(weak_pairs)}/{len(others)}")

            for other_name, score in weak_pairs:
                print(f"       ↳ {other_name}: {score:.3f}")

            print()

    # ---------- overall stats ----------
    avg_sim, median_sim, min_sim, max_sim, all_pairs = compute_similarity_stats(similarity)

    print("\n=== Overall Identity Consistency ===")
    print(f"Total comparisons: {len(all_pairs)}")
    print(f"Average similarity: {avg_sim:.3f}")
    print(f"Median similarity:  {median_sim:.3f}")
    print(f"Lowest pair:        {min_sim:.3f}")
    print(f"Highest pair:       {max_sim:.3f}")

    # ---------- weak pair summary ----------
    if weak_pairs_global:
        print(f"\n=== Weak Pair Summary (< {THRESHOLD}) ===")
        print(f"Total weak pairs: {len(weak_pairs_global)}\n")

        weak_pairs_global.sort(key=lambda x: x[2])

        for a, b, score in weak_pairs_global:
            print(f"⚠️  {a}  ↔  {b}   = {score:.3f}")
    else:
        print("\n✅ No weak identity pairs detected.")

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

    similarity = cosine_similarity(embeddings)

    display_stability_warnings(similarity, image_names)

    compute_similarity_stats(similarity)

    print("\nComputing similarity matrix...\n")

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