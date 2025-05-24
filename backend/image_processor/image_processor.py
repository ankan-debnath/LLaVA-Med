import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import networkx as nx
from PIL import Image

# Display the image
def display_image(img, image_path):
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.title(f"Image: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

# Function to load and display a single image
def load_and_display_image(image_path):
    """
    Load and display a single image

    Args:
        image_path: Path to the image

    Returns:
        np.array or None: Loaded image as a numpy array, or None if failed
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# Function to check if an image is H&E-stained
def is_he_stained(img, visualize=False):
    """
    Determine if an image is H&E-stained pathology using color analysis

    Args:
        img: Image as numpy array (RGB)
        visualize: Whether to visualize the detection process

    Returns:
        bool: True if it's an H&E-stained pathology image, False otherwise
        dict: Additional information about the detection
    """
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Extract channels
    h = hsv[:, :, 0]  # Hue
    s = hsv[:, :, 1]  # Saturation

    # H&E staining has characteristic colors:
    # - Hematoxylin: blue/purple (Hue around 120-170 in OpenCV's HSV)
    # - Eosin: pink/red (Hue around 0-20 or 160-180 in OpenCV's HSV)

    # Create masks for hematoxylin (blue/purple) and eosin (pink/red)
    hematoxylin_mask = cv2.inRange(hsv, (120, 50, 50), (170, 255, 255))
    eosin_mask1 = cv2.inRange(hsv, (0, 50, 50), (20, 255, 255))
    eosin_mask2 = cv2.inRange(hsv, (160, 50, 50), (180, 255, 255))
    eosin_mask = cv2.bitwise_or(eosin_mask1, eosin_mask2)

    # Calculate the percentage of pixels that match each stain
    total_pixels = img.shape[0] * img.shape[1]
    hematoxylin_ratio = np.sum(hematoxylin_mask > 0) / total_pixels
    eosin_ratio = np.sum(eosin_mask > 0) / total_pixels

    # Check color distribution in LAB space (more perceptually uniform)
    a = lab[:, :, 1]  # Green-Red
    b = lab[:, :, 2]  # Blue-Yellow

    # Calculate color statistics
    a_mean, a_std = np.mean(a), np.std(a)
    b_mean, b_std = np.mean(b), np.std(b)

    # H&E images typically have:
    # - Higher standard deviation in a and b (varied colors)
    # - Specific ranges of mean values in a and b
    color_variance_score = (a_std + b_std) / 2

    # Calculate texture features
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    texture_score = np.std(gray)  # Simple texture measure

    # Combine features for final decision
    is_he = (
            (hematoxylin_ratio > 0.05 or eosin_ratio > 0.1) and  # Must have some H&E colors
            color_variance_score > 10 and  # Must have color variation
            texture_score > 20  # Must have texture (not a blank slide)
    )

    # Additional information for debugging and visualization
    info = {
        'hematoxylin_ratio': hematoxylin_ratio,
        'eosin_ratio': eosin_ratio,
        'color_variance_score': color_variance_score,
        'texture_score': texture_score,
        'hematoxylin_mask': hematoxylin_mask,
        'eosin_mask': eosin_mask
    }

    # Visualize the detection process if requested
    if visualize:
        plt.figure(figsize=(15, 10))

        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')

        # Hematoxylin mask
        plt.subplot(2, 3, 2)
        plt.imshow(hematoxylin_mask, cmap='Blues')
        plt.title(f"Hematoxylin (Blue/Purple): {hematoxylin_ratio:.1%}")
        plt.axis('off')

        # Eosin mask
        plt.subplot(2, 3, 3)
        plt.imshow(eosin_mask, cmap='Reds')
        plt.title(f"Eosin (Pink/Red): {eosin_ratio:.1%}")
        plt.axis('off')

        # Combined mask
        combined_mask = cv2.bitwise_or(hematoxylin_mask, eosin_mask)
        plt.subplot(2, 3, 4)
        plt.imshow(combined_mask, cmap='gray')
        plt.title(f"Combined H&E Mask: {(hematoxylin_ratio + eosin_ratio):.1%}")
        plt.axis('off')

        # Color distribution in LAB space
        plt.subplot(2, 3, 5)
        plt.hist2d(a.flatten(), b.flatten(), bins=50, cmap='viridis')
        plt.colorbar(label='Pixel Count')
        plt.title(f"Color Distribution (LAB)\nVariance: {color_variance_score:.1f}")
        plt.xlabel('a (Green-Red)')
        plt.ylabel('b (Blue-Yellow)')

        # Texture visualization
        plt.subplot(2, 3, 6)
        edges = cv2.Canny(gray, 100, 200)
        plt.imshow(edges, cmap='gray')
        plt.title(f"Edge Detection\nTexture Score: {texture_score:.1f}")
        plt.axis('off')

        plt.suptitle(f"H&E Stain Detection: {'Positive' if is_he else 'Negative'}", fontsize=16)
        plt.tight_layout()
        plt.show()

    return is_he, info


# Function to detect nuclei in H&E-stained images
def detect_nuclei(img, visualize=False):
    """
    Detect nuclei in an H&E-stained image with high accuracy

    Args:
        img: Image as numpy array (RGB)
        visualize: Whether to visualize the detection process

    Returns:
        list: List of (x, y) coordinates for each nucleus
        int: Number of nuclei detected
        numpy.ndarray: Visualization of detected nuclei (if visualize=True, else None)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE for better contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Use a combination of thresholding methods for better results
    # 1. Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 2. Otsu's thresholding
    _, otsu_thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine the two thresholds
    binary = cv2.bitwise_and(adaptive_thresh, otsu_thresh)

    # Morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Distance transform to separate touching nuclei
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Find connected components (nuclei)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)

    # Filter nuclei based on size and shape
    nuclei_centers = []
    filtered_labels = np.zeros_like(labels, dtype=np.uint8)

    # Skip the first label (background)
    for i in range(1, num_labels):
        # Get area and dimensions
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # Filter by size (typical nuclei size in pixels)
        if 30 < area < 1000 and width < 100 and height < 100:
            # Calculate circularity
            mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = contours[0]
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    # Filter by circularity (nuclei are somewhat circular)
                    if circularity > 0.3:  # More permissive threshold
                        cx, cy = int(centroids[i][0]), int(centroids[i][1])
                        nuclei_centers.append((cx, cy))
                        filtered_labels[labels == i] = 255

    # Create visualization if requested
    vis_img = None
    if visualize:
        # Create a copy of the original image for visualization
        vis_img = img.copy()

        # Draw detected nuclei
        for cx, cy in nuclei_centers:
            cv2.circle(vis_img, (cx, cy), 5, (0, 255, 0), -1)

        # Display the detection process
        plt.figure(figsize=(20, 10))

        plt.subplot(2, 4, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(2, 4, 2)
        plt.imshow(enhanced, cmap='gray')
        plt.title("CLAHE Enhanced")
        plt.axis('off')

        plt.subplot(2, 4, 3)
        plt.imshow(adaptive_thresh, cmap='gray')
        plt.title("Adaptive Threshold")
        plt.axis('off')

        plt.subplot(2, 4, 4)
        plt.imshow(otsu_thresh, cmap='gray')
        plt.title("Otsu Threshold")
        plt.axis('off')

        plt.subplot(2, 4, 5)
        plt.imshow(binary, cmap='gray')
        plt.title("Combined Binary")
        plt.axis('off')

        plt.subplot(2, 4, 6)
        plt.imshow(dist_transform, cmap='hot')
        plt.title("Distance Transform")
        plt.axis('off')

        plt.subplot(2, 4, 7)
        plt.imshow(filtered_labels, cmap='gray')
        plt.title("Filtered Nuclei")
        plt.axis('off')

        plt.subplot(2, 4, 8)
        plt.imshow(vis_img)
        plt.title(f"Detected Nuclei: {len(nuclei_centers)}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return nuclei_centers, len(nuclei_centers), vis_img


def extract_nuclei_features(img):
    """
    Extract nuclei features for the entire image using detect_nuclei.

    Args:
        img: Image as numpy array (RGB).

    Returns:
        dict: Extracted features, including nuclei count and positions.
    """
    # Step 1: Detect nuclei using the existing function
    nuclei_centers, nuclei_count, _ = detect_nuclei(img, visualize=False)
    print(f"Detected {nuclei_count} nuclei.")

    # Step 2: Store extracted features
    features = {
        'nuclei_count': nuclei_count,
        'nuclei_centers': nuclei_centers  # List of (x, y) coordinates for nuclei
    }

    return features


def build_tissue_graph(nuclei_centers):
    """
    Build a tissue graph based on nuclei positions.

    Args:
        nuclei_centers: List of (x, y) coordinates for detected nuclei.

    Returns:
        networkx.Graph: Graph with nuclei as nodes and edges connecting nearest neighbors.
        dict: Graph metrics/features.
    """
    # Step 1: Create an empty graph
    G = nx.Graph()

    # Add nuclei as nodes
    for i, (x, y) in enumerate(nuclei_centers):
        G.add_node(i, pos=(x, y))

    # Add edges between each nucleus and its k-nearest neighbors (e.g., k=5)
    k = 5
    nuclei_array = np.array(nuclei_centers)
    for i, (x, y) in enumerate(nuclei_centers):
        distances = np.sqrt(np.sum((nuclei_array - np.array([x, y])) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[1:k + 1]  # Exclude self (distance = 0)

        for j in nearest_indices:
            G.add_edge(i, j, weight=distances[j])

    # Compute graph features
    features = {
        'num_nuclei': len(G.nodes),
        'num_edges': len(G.edges),
        'avg_degree': np.mean([d for _, d in G.degree()]),
        'clustering_coefficient': nx.average_clustering(G),
        'graph_density': nx.density(G),
        'connected_components': nx.number_connected_components(G)
    }

    return G, features


def extract_patches(img, G, overlap=0.2):
    """
    Extract exactly 9 patches with 20% overlap from an image, based on the tissue graph.

    Args:
        img: Image as numpy array (RGB).
        G: Tissue graph with nuclei as nodes and edges connecting neighbors.
        overlap: Overlap percentage between adjacent patches.

    Returns:
        list: Extracted patches as numpy arrays.
        list: Graph-based metrics for the patches (e.g., nuclei count).
    """
    h, w, _ = img.shape

    # Calculate patch sizes for a 3x3 grid with overlap
    patch_size_x = int(w / 3 + w * overlap / 3)  # Width of each patch
    patch_size_y = int(h / 3 + h * overlap / 3)  # Height of each patch

    # Stride ensures 20% overlap
    stride_x = int((w - patch_size_x) / 2)  # Two patches horizontally
    stride_y = int((h - patch_size_y) / 2)  # Two patches vertically

    patches = []
    patch_metrics = []

    # Step 1: Divide the image into exactly 9 patches (3x3 grid)
    for row in range(3):  # 3 rows
        for col in range(3):  # 3 columns
            # Calculate the top-left corner of the patch
            x_start = col * stride_x
            y_start = row * stride_y

            # Make sure patches fit within the image dimensions
            x_end = min(x_start + patch_size_x, w)
            y_end = min(y_start + patch_size_y, h)
            x_start = max(0, x_end - patch_size_x)  # Adjust start if patch goes out of bounds
            y_start = max(0, y_end - patch_size_y)

            # Extract patch
            patch = img[y_start:y_end, x_start:x_end]

            # Find nuclei within this patch
            patch_nuclei = [node for node in G.nodes if
                            x_start <= G.nodes[node]['pos'][0] < x_end and y_start <= G.nodes[node]['pos'][1] < y_end]

            # Subgraph for the patch
            patch_graph = G.subgraph(patch_nuclei)

            # Compute nuclei count for now (additional metrics can be added later)
            num_nuclei = len(patch_graph.nodes)

            # Store patch and nuclei count
            patches.append(patch)
            patch_metrics.append({
                'num_nuclei': num_nuclei
            })

    # Print results for debugging
    print("\nExtracted Patches and Their Metrics:")
    for idx, metrics in enumerate(patch_metrics):
        print(f"  Patch #{idx + 1}: {metrics}")

    return patches, patch_metrics


def rank_and_select_top_patches(patches, patch_metrics, num_top_patches=3):
    """
    Rank patches based on nuclei count and density, and select the top-ranked ones.

    Args:
        patches: List of extracted patches as numpy arrays.
        patch_metrics: List of dictionaries containing metrics for each patch.
        num_top_patches: Number of top-ranked patches to select (default is 3).

    Returns:
        list: Top-ranked patches as numpy arrays.
        list: Metrics for the top-ranked patches.
    """
    # Step 1: Rank patches based on nuclei count (primary) and density (secondary)
    ranked_indices = sorted(range(len(patches)), key=lambda i: (
        patch_metrics[i]['num_nuclei'],  # Primary: Nuclei count
        patch_metrics[i].get('graph_density', 0)  # Secondary: Graph density (default 0 if missing)
    ), reverse=True)

    # Step 2: Select top-ranked patches
    top_indices = ranked_indices[:num_top_patches]
    top_patches = [patches[i] for i in top_indices]
    top_patch_metrics = [patch_metrics[i] for i in top_indices]

    # Print top patch metrics for debugging
    print("\nTop Ranked Patches and Their Metrics:")
    for idx, metrics in enumerate(top_patch_metrics):
        print(f"  Patch #{idx + 1}: {metrics}")

    return top_patches, top_patch_metrics

def print_graph(G, image):
    # Visualize the graph alongside the original image
    print("\nVisualizing tissue graph with the original image...\n")
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Plot the original image
    axs[0].imshow(image)
    axs[0].set_title(f"Original Image")
    axs[0].axis('off')

    # Plot the tissue graph
    pos = nx.get_node_attributes(G, 'pos')  # Get node positions
    axs[1].imshow(image, alpha=0.6)  # Overlay graph on the image
    nx.draw(G, pos, node_size=40, node_color='red', edge_color='blue', with_labels=False, ax=axs[1])
    axs[1].set_title(f"Tissue Graph for Image")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    print("\n" + "-" * 50 + "\n")

def print_patches(patches):
    # Visualize the top 3 patches
    print("\nVisualizing the top 3 patches...\n")
    fig, axes = plt.subplots(1, len(patches), figsize=(15, 5))
    for j, patch in enumerate(patches):
        axes[j].imshow(patch)
        axes[j].set_title(f"Patch #{j + 1}")
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


def save_top_patches(top_patches, save_dir='temp_uploads', prefix='patch'):
    """
    Save a list of image patches to a directory.

    Parameters:
    - top_patches: list of numpy arrays representing images.
    - save_dir: folder where images will be saved.
    - prefix: filename prefix for each patch image.
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, patch in enumerate(top_patches):
        # Convert to PIL image
        img = Image.fromarray(patch)

        # Create filename
        filename = f"{prefix}_{i + 1}.png"
        filepath = os.path.join(save_dir, filename)

        # Save image
        img.save(filepath)

    print(f"✅ Saved {len(top_patches)} patches to '{save_dir}'")

def analyze_hne_image(image_path):
    # Step 1: Load Image
    image = load_and_display_image(image_path)
    if image is None:
        print("Failed to load image.")
        return

    # Step 2: Check H&E stain
    if not is_he_stained(image):
        print("Image is not H&E stained.")
        return

    # Step 3: Detect Nuclei
    print("Detecting nuclei...")
    features  = extract_nuclei_features(image)


    # Step 4: Build Tissue Graph
    print("Building tissue graph...")
    nuclei_centers = features['nuclei_centers']
    G, graph_features = build_tissue_graph(nuclei_centers)
    # print_graph(G, image)

    # Step 5: Extract Patches
    print("Extracting patches...")
    patches, patch_metrics = extract_patches(image, G, overlap=0.2)

    # Step 6 : Rank and select the top 3 patches
    top_patches, top_patch_metrics = rank_and_select_top_patches(patches, patch_metrics, num_top_patches=3)
    save_top_patches(top_patches)

    return top_patches

# image_path = "img_102.jpg"
# patches = analyze_hne_image(image_path)


