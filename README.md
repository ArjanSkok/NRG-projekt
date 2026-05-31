Object Segmentation from Multi-View LiDAR Renderings



Interactive object segmentation for aerial LiDAR point clouds using SAM2 and multiple 2D renderings.



The user selects one point on a target object in a 3D point cloud. The system then renders the point cloud from multiple virtual camera views, segments the object in each 2D view with SAM2, projects the masks back into 3D, and fuses the results with normalized voting.



\## Usage



\### 1. Prepare data



Place a `.laz` file in the `data/` folder. There are already some example data there, that was used for development.



Example:



```text

data/skocjan.laz



Then set the input file path in `sam2\_segment.py`:



`filename\_base = "./data/skocjan.laz"`



\### 2. Prepare SAM2



Download a SAM2 checkpoint separately from the official SAM2 repository.



Then set the checkpoint and config paths in `sam2\_segment.py`:



`SAM2\_CHECKPOINT = r"<path\_to\_sam2\_folder>\\sam2\\checkpoints\\sam2.1\_hiera\_large.pt"`



`SAM2\_CONFIG = "configs/sam2.1/sam2.1\_hiera\_l.yaml"`



The SAM2 checkpoint is not included in this repository.



\### 3. Install dependencies



Install the Python dependencies:



`pip install -r requirements.txt`



SAM2 may need to be installed separately from the official repository:



`git clone https://github.com/facebookresearch/sam2.git`



`cd sam2`



`pip install -e .`



\### 4. Run interactive segmentation



Run:



`python main.py`



A 3D point cloud window opens.



Controls:



`Shift + Left Click` — select a point on the target object



`Q / Esc` — confirm selection and close the picker



After the point is selected, the system automatically renders the LiDAR crop from multiple 2D views, runs SAM2 on the rendered views, back-projects the 2D masks into 3D, fuses the masks with normalized voting, and opens the final 3D segmentation result.



\### 5. Outputs



Generated files are saved into:



`renders/` — generated 2D height renderings and point-index maps



`masks/` — SAM2 masks for individual views



`viz/` — final visualizations



Important output files:



`viz/all\_masks.png` — SAM2 masks over all rendered views



`viz/segmentation\_topdown.png` — top-down visualization of the final segment



`viz/segmented\_result.ply` — colored 3D point cloud result

