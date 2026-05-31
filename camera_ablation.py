import laspy
import numpy as np
import os
import sam2_segment as seg
import matplotlib.pyplot as plt
import shutil

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

filename = "./data/vojkovo.laz"
crop_size = 150

las = laspy.read(filename)
x, y, z = las.x, las.y, las.z
cls = np.array(las.classification)

cx = (x.min() + x.max()) / 2
cy = (y.min() + y.max()) / 2

crop_mask = (
    (x > cx - crop_size/2) & (x < cx + crop_size/2) &
    (y > cy - crop_size/2) & (y < cy + crop_size/2) &
    (~np.isin(cls, [7, 12]))
)

xyz_crop = np.vstack([x, y, z]).T[crop_mask]
cls_crop = cls[crop_mask]
mean = xyz_crop.mean(axis=0)
xyz_norm = xyz_crop - mean
centroid = xyz_norm.mean(axis=0)

#chosen_3d = np.load("chosen_point.npy")
chosen_3d = [53, -35, 2.1]

device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2(seg.SAM2_CONFIG, seg.SAM2_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)

results = []
for n_views in [0, 2, 4, 8, 16]:
    print(f"\nn_views={n_views}")
    for folder in ["renders", "masks"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    cameras = seg.get_cameras(centroid, xyz_norm, n_views=n_views)
    seg.render_views(xyz_norm, centroid, cameras)

    top_name, top_cam = cameras[-1]
    R_top = seg.rotation_matrix(top_cam, np.zeros(3))
    top_prompt_pixel = seg.project_single_point(chosen_3d, top_cam, R_top)

    masks, _ = seg.segment_pics(predictor, cameras, xyz_norm,
                                 top_prompt_point=np.array([top_prompt_pixel]))

    votes, accepted, segmented_pts = seg.voting(cameras, xyz_norm)
    target_centroid = xyz_norm[accepted].mean(axis=0)
    majority_class, precision, recall, f1 = seg.evaluate(accepted, cls_crop, xyz_norm, target_centroid, radius=15)

    if n_views == 8:
        cameras_8 = cameras

    results.append({"n_views": n_views, "precision": precision, "recall": recall, "f1": f1, "points": accepted.sum()})

print("\n=== ABLATION RESULTS ===")
print(f"{'Views':>6} {'Precision':>10} {'Points':>8}")
for r in results:
    print(f"{r['n_views']:>6} {r['precision']:>10.3f} {r['points']:>8}")

views = [r["n_views"] for r in results]
precision = [r["precision"] for r in results]
recall = [r["recall"] for r in results]
f1 = [r["f1"] for r in results]
points = [r["points"] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(views, precision, 'o-', label="Precision")
ax1.plot(views, recall, 'o-', label="Recall")
ax1.plot(views, f1, 'o-', label="F1")
ax1.set_xlabel("Število stranskih pogledov")
ax1.set_ylabel("Vrednost metrike")
ax1.set_title("Metrike glede na število pogledov")
ax1.set_xticks(views)
ax1.grid(True)
ax1.legend()

ax2.plot(views, points, 'o-')
ax2.set_xlabel("Število stranskih pogledov")
ax2.set_ylabel("Število točk")
ax2.set_title("Segmentirane točke glede na število pogledov")
ax2.set_xticks(views)
ax2.grid(True)

plt.tight_layout()
plt.savefig("ablation_views.png", dpi=150)
plt.show()

print("\nTHRESHOLD ABLATION (n_views=8 side + top)")
results_thresh = []

cameras = cameras_8

for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    votes, accepted, segmented_pts = seg.voting(cameras, xyz_norm, prob_threshold=thresh)
    if accepted.sum() == 0:
        print(f"  threshold={thresh:.1f}: no points selected")
        results_thresh.append({"threshold": thresh, "precision": 0.0, "recall": 0.0, "f1": 0.0, "points": 0})
        continue

    target_centroid = xyz_norm[accepted].mean(axis=0)

    majority_class, precision, recall, f1 = seg.evaluate(accepted, cls_crop, xyz_norm, target_centroid, radius=15)

    results_thresh.append({"threshold": thresh, "precision": precision, "recall": recall, "f1": f1, "points": accepted.sum()})

    print(f"threshold={thresh:.1f}: " f"precision={precision:.3f}, "f"recall={recall:.3f}, "f"f1={f1:.3f}, "f"points={accepted.sum()}")


thresholds = [r["threshold"] for r in results_thresh]
precisions = [r["precision"] for r in results_thresh]
recalls = [r["recall"] for r in results_thresh]
f1s = [r["f1"] for r in results_thresh]
points_t = [r["points"] for r in results_thresh]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(thresholds, precisions, 'o-', label="Precision")
ax1.plot(thresholds, recalls, 'o-', label="Recall")
ax1.plot(thresholds, f1s, 'o-', label="F1")
ax1.set_xlabel("Prag verjetnosti")
ax1.set_ylabel("Vrednost metrike")
ax1.set_title("Metrike glede na prag glasovanja")
ax1.grid(True)
ax1.legend()

ax2.plot(thresholds, points_t, 'o-')
ax2.set_xlabel("Prag verjetnosti")
ax2.set_ylabel("Število točk")
ax2.set_title("Segmentirane točke glede na prag glasovanja")
ax2.grid(True)

plt.tight_layout()
plt.savefig("ablation_threshold.png", dpi=150)
plt.show()