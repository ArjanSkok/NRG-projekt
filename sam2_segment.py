import laspy
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import matplotlib.pyplot as plt
import os
from scipy.ndimage import maximum_filter

filename_base = "./data/skocjan.laz"
crop_size = 150

SAM2_CHECKPOINT = r"C:\FRI\Magisterij\NRG\sam2\checkpoints\sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

def get_cameras(centroid, xyz_norm, n_views=8):
    scene_size = np.max([xyz_norm[:, 0].max() - xyz_norm[:, 0].min(), xyz_norm[:, 1].max() - xyz_norm[:, 1].min()]) / 2
    camera_r = scene_size * 1.8
    camera_h  = scene_size * 1.2

    cameras = []
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        camera_position = centroid + np.array([camera_r * np.cos(angle), camera_r * np.sin(angle), camera_h])
        cameras.append((i, camera_position))
    cameras.append(("top", centroid + np.array([0, 0, camera_r * 1.2])))

    return cameras

def rotation_matrix(camera_position, target, up=np.array([0,0,1])):
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    return np.stack([right, up, forward], axis=0)

def project_points(xyz, camera_position, R, fx=800, img_w=1024, img_h=1024):
    camera_space = (xyz - camera_position) @ R.T
    depth = camera_space[:, 2]
    mask = depth > 0.5
    u = (camera_space[:, 0] / depth * fx + img_w / 2).astype(np.float32)
    v = (-camera_space[:, 1] / depth * fx + img_h / 2).astype(np.float32)
    mask &= (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    return u, v, depth, mask

def render_heightmap(xyz, camera_position, R, img_w=1024, img_h=1024, fx=800, point_radius=2):
    u, v, depth, mask = project_points(xyz, camera_position, R, fx, img_w, img_h)
    valid_indices = np.where(mask)[0]
    u_v = u[mask].astype(np.int32)
    v_v = v[mask].astype(np.int32)
    depth_v = depth[mask]
    z_v = xyz[mask, 2]

    depth_buf = np.full((img_h, img_w), np.inf)
    height_buf = np.full((img_h, img_w), np.nan)
    point_index_image = -np.ones((img_h, img_w), dtype=np.int32)

    order = np.argsort(depth_v)
    for j in order:
        px, py = u_v[j], v_v[j]
        if depth_v[j] < depth_buf[py, px]:
            depth_buf[py, px] = depth_v[j]
            height_buf[py, px] = z_v[j]
            point_index_image[py, px] = valid_indices[j]

    filled = ~np.isnan(height_buf)
    img_gray = np.zeros((img_h, img_w), dtype=np.uint8)
    if np.any(filled):
        h = height_buf[filled]
        h_min, h_max = np.percentile(h, [5, 95])
        h_norm = np.zeros((img_h, img_w), dtype=np.float32)
        h_norm[filled] = np.clip((height_buf[filled] - h_min) / (h_max - h_min + 1e-6), 0, 1)
        img_gray[filled] = (h_norm[filled] * 255).astype(np.uint8)

    if point_radius > 0:
        img_gray = maximum_filter(img_gray, size=2 * point_radius + 1)

    return np.stack([img_gray]*3, axis=-1), point_index_image

def render_views(xyz_norm, centroid, cameras):
    for name, camera_position in cameras:
        R = rotation_matrix(camera_position, np.zeros(3))
        img, point_index_image = render_heightmap(xyz_norm, camera_position, R)
        Image.fromarray(img).save(f"renders/{name}.png")
        np.save(f"renders/{name}_indices.npy", point_index_image)


def project_single_point(centroid_3d, camera_position, R, fx=800, img_w=1024, img_h=1024):
    point_cam = R @ (centroid_3d - camera_position)
    depth = point_cam[2]

    if depth <= 0:
        return None
    
    u = int(point_cam[0] / depth * fx + img_w / 2)
    v = int(-point_cam[1] / depth * fx + img_h / 2)

    if 0 <= u < img_w and 0 <= v < img_h:
        return np.array([u, v])
    
    return None

def segment_pics(predictor, cameras, xyz_norm, top_prompt_point=np.array([[512, 430]])):
    img_top = np.array(Image.open("renders/top.png").convert("RGB"))
    predictor.set_image(img_top)

    masks, scores, _ = predictor.predict(point_coords=top_prompt_point, point_labels=np.array([1]), multimask_output=True)
    masks = np.squeeze(masks).astype(bool)
    best_mask = masks[np.argmax(scores[:2])]
    top_indices = np.load("renders/top_indices.npy")
    masked_indices = top_indices[best_mask]
    masked_indices = masked_indices[masked_indices >= 0]
    building_centroid_3d = xyz_norm[masked_indices].mean(axis=0)
    np.save("masks/top_mask.npy", best_mask)

    final_masks = {"top": best_mask}
    for name, camera_position in cameras[:-1]:
        R = rotation_matrix(camera_position, np.zeros(3))
        prompt_pt = project_single_point(building_centroid_3d, camera_position, R)

        if prompt_pt is None:
            continue

        img = np.array(Image.open(f"renders/{name}.png").convert("RGB"))
        predictor.set_image(img)

        input_points = np.array([prompt_pt, [512, 512]])
        input_labels = np.array([1, 0])

        if np.linalg.norm(prompt_pt - np.array([512, 512])) < 50:
            input_points = np.array([prompt_pt])
            input_labels = np.array([1])

        masks_v, scores_v, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)
        masks_v = np.squeeze(masks_v).astype(bool)

        max_area = 1024 * 1024 * 0.4
        for i, m in enumerate(masks_v):
            if m.sum() > max_area:
                scores_v[i] = 0

        best = masks_v[np.argmax(scores_v)]
        final_masks[name] = best
        np.save(f"masks/{name}_mask.npy", best)
        print(f"  {name}: target={prompt_pt}, score={scores_v.max():.2f}, area={best.sum()}")

    return final_masks, building_centroid_3d

def voting(cameras, xyz_norm, prob_threshold=0.45):
    object_score = np.zeros(len(xyz_norm), dtype=np.float32)
    total_score = np.zeros(len(xyz_norm), dtype=np.float32)

    for name, _ in cameras:
        mask_path = f"masks/{name}_mask.npy"
        idx_path = f"renders/{name}_indices.npy"

        if not os.path.exists(mask_path):
            continue

        mask = np.load(mask_path)
        point_index_image = np.load(idx_path)

        visible = point_index_image >= 0
        visible_indices = point_index_image[visible]

        total_score[visible_indices] += 1.0

        selected = point_index_image[mask]
        selected = selected[selected >= 0]
        object_score[selected] += 1.0

    object_prob = object_score / np.maximum(total_score, 1e-6)

    accepted = object_prob >= prob_threshold
    segmented_pts = xyz_norm[accepted]

    return object_prob, accepted, segmented_pts

def evaluate(mask, cls_crop, xyz_norm, centroid=None, radius=15):
    segmented_cls = cls_crop[mask]
    class_names = {1:'unclassified', 2:'ground', 3:'low veg', 4:'med veg', 5:'high veg', 6:'building', 9:'water'}
    unique, counts = np.unique(segmented_cls, return_counts=True)

    print()
    print("Classes:")
    for c, n in zip(unique, counts):
        print(f"  class {c} ({class_names.get(c,'?')}): {n} points ({100*n/len(segmented_cls):.1f}%)")

    majority_idx = np.argmax(counts)
    majority_class = unique[majority_idx]
    majority_count = counts[majority_idx]
    precision = majority_count / len(segmented_cls)

    if centroid is not None:
        dist = np.linalg.norm(xyz_norm - centroid, axis=1)
        target_object = (cls_crop == majority_class) & (dist < radius)
        true_positives = (mask & target_object).sum()
        recall = true_positives / max(target_object.sum(), 1)
    else:
        recall = 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print()
    print(f"Majority class: {majority_class} ({class_names.get(majority_class, '?')})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")

    return majority_class, precision, recall, f1

def visualize_masks(cameras, masks):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for ax, (name, _) in zip(axes, cameras):
        img = np.array(Image.open(f"renders/{name}.png").convert("RGB"))
        ax.imshow(img, cmap="gray")
        if name in masks:
            overlay = np.zeros((*masks[name].shape, 4), dtype=np.float32)
            overlay[masks[name]] = [0.0, 1.0, 0.5, 0.6]
            ax.imshow(overlay)
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("./viz/all_masks.png", dpi=150)

def visualize_3d(xyz_norm, segmented_mask_3d, segmented_pts):
    import open3d as o3d
    colors = np.full((len(xyz_norm), 3), 0.5)
    colors[segmented_mask_3d] = [1.0, 0.2, 0.2]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_norm)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud("./viz/segmented_result.ply", pcd)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xyz_norm[~segmented_mask_3d, 0], xyz_norm[~segmented_mask_3d, 1], c='lightgrey', s=0.1)
    ax.scatter(segmented_pts[:, 0], segmented_pts[:, 1], c='red', s=2.0)
    ax.set_title(f"3D segmentation result (top-down)\n{len(segmented_pts)} points")
    ax.set_aspect('equal')
    plt.savefig("./viz/segmentation_topdown.png", dpi=150)

def main(chosen_3d=None, filename=filename_base):
    os.makedirs("renders", exist_ok=True)
    os.makedirs("masks", exist_ok=True)
    os.makedirs("viz", exist_ok=True)

    las = laspy.read(filename)
    x, y, z = las.x, las.y, las.z
    cls = np.array(las.classification)

    cx = (x.min() + x.max()) / 2
    cy = (y.min() + y.max()) / 2

    crop_mask = ((x > cx - crop_size/2) & (x < cx + crop_size/2) & (y > cy - crop_size/2) & (y < cy + crop_size/2) & (~np.isin(cls, [7, 12])))
    xyz_crop = np.vstack([x, y, z]).T[crop_mask]
    cls_crop = cls[crop_mask]

    xyz_norm = xyz_crop - xyz_crop.mean(axis=0)
    centroid = xyz_norm.mean(axis=0)

    cameras = get_cameras(centroid, xyz_norm)
    render_views(xyz_norm, centroid, cameras)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    if chosen_3d is None:
        chosen_3d = np.load("chosen_point.npy")

    _, top_camera_position = cameras[-1]
    R_top = rotation_matrix(top_camera_position, np.zeros(3))
    top_prompt_pixel = project_single_point(chosen_3d, top_camera_position, R_top)

    masks, _ = segment_pics(predictor, cameras, xyz_norm, top_prompt_point=np.array([top_prompt_pixel]))
    visualize_masks(cameras, masks)

    _, accepted, segmented_pts = voting(cameras, xyz_norm)
    target_centroid = xyz_norm[accepted].mean(axis=0)
    evaluate(accepted, cls_crop, xyz_norm, target_centroid)
    visualize_3d(xyz_norm, accepted, segmented_pts)

    return xyz_norm, cls_crop, accepted, segmented_pts

if __name__ == "__main__":
    main()