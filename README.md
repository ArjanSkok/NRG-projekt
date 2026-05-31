# Object Segmentation from Multi-View LiDAR Renderings

Interactive object segmentation on aerial LiDAR point clouds using SAM2. The system renders the point cloud from multiple virtual camera viewpoints, segments each view with SAM2, and fuses the results back into 3D space using normalized voting.

## Requirements

- Python 3.11+
- CUDA-capable GPU (recommended)

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
```

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```
**Windows CMD:**
```cmd
.\.venv\Scripts\activate.bat
```
**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install SAM2
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

### 5. Download a SAM2 checkpoint
Download from the [official SAM2 repository](https://github.com/facebookresearch/sam2).

### 6. Configure paths
In `sam2_segment.py`, set:
```python
SAM2_CHECKPOINT = "path/to/your/checkpoint.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
filename_base = "./data/your_file.laz"
```

### 7. Add LiDAR data
Place `.laz` files into the `data/` folder. Sample datasets used during development are already included.

## Usage

```bash
python main.py
```

A 3D point cloud viewer opens. Select the target object:

| Action | Key |
|--------|-----|
| Pick point | `Shift + Left Click` |
| Confirm and close | `Q` or `Esc` |

The system will then automatically:
1. Render the point cloud from 9 virtual camera views
2. Segment each view with SAM2
3. Back-project masks into 3D space
4. Fuse results with normalized voting
5. Display the final 3D segmentation

## Ablation

To reproduce the ablation experiments from the paper:
```bash
python ablation.py
```

This generates:
- `ablation_views.png` — effect of number of views on segmentation quality
- `ablation_threshold.png` — effect of voting threshold on precision/recall/F1
