Image Clustering with DINOv2 and HDBSCAN
========================================

### About the Project

This Python project leverages the **DINOv2** model to create high-dimensional embeddings for images randomly selected from a ZIP file. It then uses the **HDBSCAN** algorithm to cluster these embeddings. The final output includes a well-organized directory of clustered images, a detailed report file, and a 2D visualization of the clustering results using PCA.

### Installation

To get started, install the required libraries by running the following command:

Bash

```
pip install torch numpy matplotlib scikit-learn hdbscan transformers Pillow

```

### How to Use

1.  **Prepare your ZIP file:** Create a ZIP file named `images.zip` containing all the images you want to cluster. You can organize your images into subfolders within the ZIP file if you want, but it's not strictly necessary.

2.  **Run the script:** Simply execute the Python script. It will automatically handle the entire process for you: selecting images from the ZIP, generating embeddings, performing the clustering, and saving all the outputs.

Bash

```
python image_clusterer.py

```

### Outputs

Once the script finishes, you will find the following files and folders in your project directory:

-   `images/`: A copy of all the images that were selected from your ZIP file.

-   `clusters/`: This is where your clustered images are saved. Each cluster has its own folder named `cluster_X` (where `X` is the cluster ID). Images that are not assigned to a specific cluster (noise) are placed in the `noise` folder.

-   `cluster_report.txt`: A text file that lists each image's filename and its assigned cluster ID.

-   `DINOv2_HDBSCAN_Clustering.png`: A scatter plot that visualizes the clusters. The high-dimensional embeddings are reduced to two dimensions using **PCA** (Principal Component Analysis) for this plot.
