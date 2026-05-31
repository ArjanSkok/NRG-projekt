import open3d
import numpy as np
import laspy
import sam2_segment
import sam_segment

filename = "./data/vojkovo.laz"
crop_size = 150

def class_colors(cls_crop):
    colors = np.full((len(cls_crop), 3), 0.6)

    colors[cls_crop == 2] = [0.5, 0.35, 0.2]
    colors[cls_crop == 3] = [0.3, 0.8, 0.3]
    colors[cls_crop == 4] = [0.1, 0.6, 0.1]
    colors[cls_crop == 5] = [0.0, 0.4, 0.0]
    colors[cls_crop == 6] = [1.0, 0.0, 0.0]
    colors[cls_crop == 9] = [0.0, 0.0, 1.0]

    return colors

def load_crop():
    las = laspy.read(filename)
    x, y, z = las.x, las.y, las.z
    cls = np.array(las.classification)

    cx = (x.min() + x.max()) / 2
    cy = (y.min() + y.max()) / 2

    crop_mask = ((x > cx - crop_size/2) & (x < cx + crop_size/2) & (y > cy - crop_size/2) & (y < cy + crop_size/2) & (~np.isin(cls, [7, 12])))

    xyz_crop = np.vstack([x, y, z]).T[crop_mask]
    cls_crop = cls[crop_mask]

    mean = xyz_crop.mean(axis=0)
    xyz_norm = xyz_crop - mean

    return xyz_norm, cls_crop


def pick_point(xyz_norm, cls_crop):
    colors = class_colors(cls_crop)

    pointcloud = open3d.geometry.PointCloud()
    pointcloud.points = open3d.utility.Vector3dVector(xyz_norm)
    pointcloud.colors = open3d.utility.Vector3dVector(colors)

    print("=== INSTRUCTIONS ===")
    print("Shift + Left Click  — pick a point on the target building")
    print("Q or Escape         — confirm and exit")
    print("====================")

    vis = open3d.visualization.VisualizerWithEditing()
    vis.create_window("Pick target object")
    vis.add_geometry(pointcloud)
    vis.run()
    vis.destroy_window()

    picked_indices = vis.get_picked_points()

    if len(picked_indices) == 0:
        return None

    chosen_3d = xyz_norm[picked_indices[-1]]

    return chosen_3d


def show_result(xyz_norm, cls_crop, accepted):
    colors = class_colors(cls_crop)
    colors[accepted] = [0.0, 0.8, 1.0]

    pointcloud = open3d.geometry.PointCloud()
    pointcloud.points = open3d.utility.Vector3dVector(xyz_norm)
    pointcloud.colors = open3d.utility.Vector3dVector(colors)

    open3d.visualization.draw_geometries([pointcloud], window_name="Segmentation result")


def main():
    xyz_norm, cls_crop = load_crop()

    chosen_3d = pick_point(xyz_norm, cls_crop)
    if chosen_3d is None:
        print("No point picked.")
        return

    xyz_norm, cls_crop, accepted, segmented_pts = sam2_segment.main(chosen_3d, filename)
    show_result(xyz_norm, cls_crop, accepted)

if __name__ == "__main__":
    main()