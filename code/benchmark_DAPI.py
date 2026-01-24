import cv2
import gc
import numpy as np
import os
import pandas as pd
import tifffile
from scipy import ndimage
from matplotlib import pyplot as plt


# File paths
dataset = "MERSCOPE_WT_1"
data_path = f"../data/{dataset}/"
output_path = f"../output/{dataset}/"


# Transformation parameters
pixel_size = 0.10799861
x_shift = int(-266.1734)
y_shift = int(180.2510)


# All DAPI images
files = os.listdir(data_path + "raw_data/DAPI_images/")
files = [i for i in files if i.startswith("mosaic")]
files.sort()


# Read transcripts
transcripts = pd.read_csv(data_path + "raw_data/transcripts.csv")
transcripts = transcripts[["cell_id", "gene", "global_x", "global_y", "global_z"]].copy()


# Define target genes
all_genes = pd.read_csv(data_path + "processed_data/genes.csv")
all_genes = all_genes["genes"].tolist()

granule_markers = ["Camk2a", "Cplx2", "Slc17a7", "Ddn", "Syp", "Map1a", "Shank1", "Syn1", "Gria1", "Gria2", "Cyfip2", "Vamp2", "Bsn", "Slc32a1", "Nfasc", "Syt1", "Tubb3", "Nav1", "Shank3", "Mapt"]

nc_markers = pd.read_csv(data_path + "processed_data/negative_controls.csv")
nc_markers = nc_markers["Gene"].tolist()


# ========================= Initial Nuclei Mask Analysis (No Dilation) =========================#

# Read all DAPI images from different z-layers (memory-efficient incremental processing)
print(f"Reading all DAPI images from {len(files)} z-layers...\n")

# First pass: get image dimensions
print("First pass: Getting image dimensions...")
first_img_path = os.path.join(data_path, "raw_data/DAPI_images", files[0])
first_img = tifffile.imread(first_img_path)
first_img = cv2.normalize(first_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
height, width = first_img.shape
print(f"Image shape: {height} x {width}\n")

# Initialize accumulators for memory-efficient processing
# non_zero_count = np.zeros((height, width), dtype=np.uint8)  # Commented out - only used for "at least N layers" strategies
mip_accumulator = first_img.copy().astype(np.float32)
mean_accumulator = first_img.copy().astype(np.float32)
median_list = [first_img.astype(np.float32)]

# Process first image
# th_first = cv2.adaptiveThreshold(first_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, -1)  # Commented out - only used for "at least N layers" strategies
# non_zero_count += (th_first > 0).astype(np.uint8)  # Commented out - only used for "at least N layers" strategies

# Process remaining images incrementally (one at a time to save memory)
print("Processing images incrementally to save memory...")
for i, fname in enumerate(files[1:], start=1):
    print(f"  Processing layer {i+1}/{len(files)}: {fname}")
    img_path = os.path.join(data_path, "raw_data/DAPI_images", fname)
    img = tifffile.imread(img_path)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Update accumulators
    # th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, -1)  # Commented out - only used for "at least N layers" strategies
    # non_zero_count += (th > 0).astype(np.uint8)  # Commented out - only used for "at least N layers" strategies
    mip_accumulator = np.maximum(mip_accumulator, img.astype(np.float32))
    mean_accumulator += img.astype(np.float32)
    median_list.append(img.astype(np.float32))
    
    # Free memory immediately
    del img
    if i % 2 == 0:  # Force garbage collection every 2 images
        gc.collect()

# Finalize accumulators
mean_accumulator /= len(files)
mip_accumulator = mip_accumulator.astype(np.uint8)
mean_accumulator = mean_accumulator.astype(np.uint8)

# Compute median (still requires stacking, but only for median)
print("Computing median projection...")
median_accumulator = np.median(np.stack(median_list, axis=0), axis=0).astype(np.uint8)
del median_list, first_img
gc.collect()

print(f"Completed processing all {len(files)} images\n")
num_layers = len(files)

# DoG filter parameters (from manuscript: σ1=0, σ2=20 pixels)
DOG_SIGMA = 20.0  # Standard deviation for negative Gaussian (positive Gaussian has σ=0, i.e., original image)

def apply_dog_filter(image, sigma=DOG_SIGMA):
    """
    Apply Difference of Gaussians (DoG) filter.
    From manuscript: positive Gaussian with σ=0 (original image) minus 
    negative Gaussian with σ=20 pixels.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (uint8)
    sigma : float
        Standard deviation for the negative Gaussian (default: 20.0)
    
    Returns:
    --------
    np.ndarray
        DoG filtered image (uint8)
    """
    # Convert to float for processing
    img_float = image.astype(np.float32)
    
    # Apply Gaussian blur (negative Gaussian with σ=sigma)
    blurred = ndimage.gaussian_filter(img_float, sigma=sigma)
    
    # DoG = original (σ=0) - blurred (σ=sigma)
    dog_result = img_float - blurred
    
    # Normalize back to uint8 range
    dog_result = cv2.normalize(dog_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return dog_result

# Define stacking strategies (now using pre-computed accumulators)
# Commented out "at least N layers" strategies - thresholding before merging loses information
# strategies = {
#     "At least 1 layer": {
#         "mask": lambda: (non_zero_count >= 1).astype(np.uint8) * 255,
#         "description": "Pixel is 1 if non-zero in at least 1 layer"
#     },
#     "At least 2 layers": {
#         "mask": lambda: (non_zero_count >= 2).astype(np.uint8) * 255,
#         "description": "Pixel is 1 if non-zero in at least 2 layers"
#     },
#     "At least 3 layers": {
#         "mask": lambda: (non_zero_count >= 3).astype(np.uint8) * 255,
#         "description": "Pixel is 1 if non-zero in at least 3 layers"
#     },
#     "At least 4 layers": {
#         "mask": lambda: (non_zero_count >= 4).astype(np.uint8) * 255,
#         "description": "Pixel is 1 if non-zero in at least 4 layers"
#     },
# }

strategies = {
    "Maximum Intensity Projection (MIP)": {
        "mask": lambda: cv2.adaptiveThreshold(
            mip_accumulator,
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, -1
        ),
        "description": "Take max across layers, then threshold"
    },
    "Maximum Intensity Projection (MIP) + DoG": {
        "mask": lambda: cv2.adaptiveThreshold(
            apply_dog_filter(mip_accumulator),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, -1
        ),
        "description": "Take max across layers, apply DoG filter (σ=20), then threshold"
    },
    "Mean Intensity Projection": {
        "mask": lambda: cv2.adaptiveThreshold(
            mean_accumulator,
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, -1
        ),
        "description": "Take mean across layers, then threshold"
    },
    "Mean Intensity Projection + DoG": {
        "mask": lambda: cv2.adaptiveThreshold(
            apply_dog_filter(mean_accumulator),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, -1
        ),
        "description": "Take mean across layers, apply DoG filter (σ=20), then threshold"
    },
    "Median Intensity Projection": {
        "mask": lambda: cv2.adaptiveThreshold(
            median_accumulator,
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, -1
        ),
        "description": "Take median across layers, then threshold"
    },
    "Median Intensity Projection + DoG": {
        "mask": lambda: cv2.adaptiveThreshold(
            apply_dog_filter(median_accumulator),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, -1
        ),
        "description": "Take median across layers, apply DoG filter (σ=20), then threshold"
    }
}

# Calculate downsampling dimensions (once, since all images have same size)
if height < width:
    scale_factor = 5000 / height
    new_height = 5000
    new_width = int(width * scale_factor)
else:
    scale_factor = 5000 / width
    new_width = 5000
    new_height = int(height * scale_factor)

# Create output directory
output_dir = os.path.join(data_path, "intermediate_data")
os.makedirs(output_dir, exist_ok=True)

# Process each strategy
results = []
for strategy_name, strategy_info in strategies.items():
    print("="*80)
    print(f"STRATEGY: {strategy_name}")
    print(f"Description: {strategy_info['description']}")
    print("="*80)
    
    # Generate merged mask
    merged_mask = strategy_info["mask"]()
    
    # Detect individual nuclei masks using findContours
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate areas for each detected nucleus
    nuclei_areas_pixels = []
    nuclei_areas_um2 = []
    nuclei_radii_um = []
    
    for contour in contours:
        area_pixels = cv2.contourArea(contour)
        if area_pixels > 0:  # Filter out tiny artifacts
            nuclei_areas_pixels.append(area_pixels)
            area_um2 = area_pixels * (pixel_size ** 2)
            nuclei_areas_um2.append(area_um2)
            # Approximate radius assuming circular shape: area = π * r^2
            radius_um = np.sqrt(area_um2 / np.pi)
            nuclei_radii_um.append(radius_um)
    
    # Calculate statistics
    if len(nuclei_areas_pixels) > 0:
        print(f"Number of detected nuclei: {len(nuclei_areas_pixels)}")
        print(f"\nArea statistics (pixels²):")
        print(f"  Mean: {np.mean(nuclei_areas_pixels):.1f}")
        print(f"  Median: {np.median(nuclei_areas_pixels):.1f}")
        print(f"  Min: {np.min(nuclei_areas_pixels):.1f}")
        print(f"  Max: {np.max(nuclei_areas_pixels):.1f}")
        print(f"  Std: {np.std(nuclei_areas_pixels):.1f}")
        
        print(f"\nArea statistics (μm²):")
        print(f"  Mean: {np.mean(nuclei_areas_um2):.2f}")
        print(f"  Median: {np.median(nuclei_areas_um2):.2f}")
        print(f"  Min: {np.min(nuclei_areas_um2):.2f}")
        print(f"  Max: {np.max(nuclei_areas_um2):.2f}")
        print(f"  Std: {np.std(nuclei_areas_um2):.2f}")
        
        print(f"\nRadius statistics (μm, approximated as circle):")
        print(f"  Mean: {np.mean(nuclei_radii_um):.2f}")
        print(f"  Median: {np.median(nuclei_radii_um):.2f}")
        print(f"  Min: {np.min(nuclei_radii_um):.2f}")
        print(f"  Max: {np.max(nuclei_radii_um):.2f}")
        print(f"  Std: {np.std(nuclei_radii_um):.2f}")
        
        # Store results
        results.append({
            "strategy": strategy_name,
            "num_nuclei": len(nuclei_areas_pixels),
            "mean_area_pixels": np.mean(nuclei_areas_pixels),
            "median_area_pixels": np.median(nuclei_areas_pixels),
            "mean_area_um2": np.mean(nuclei_areas_um2),
            "median_area_um2": np.median(nuclei_areas_um2),
            "mean_radius_um": np.mean(nuclei_radii_um),
            "median_radius_um": np.median(nuclei_radii_um)
        })
    else:
        print("No nuclei detected!")
        results.append({
            "strategy": strategy_name,
            "num_nuclei": 0,
            "mean_area_pixels": 0,
            "median_area_pixels": 0,
            "mean_area_um2": 0,
            "median_area_um2": 0,
            "mean_radius_um": 0,
            "median_radius_um": 0
        })
    
    # Save downsampled merged mask
    merged_mask_downsampled = cv2.resize(merged_mask, (new_width, new_height), interpolation=cv2.INTER_AREA)
    safe_filename = strategy_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    output_path = os.path.join(output_dir, f"merged_mask_{safe_filename}_downsampled.png")
    cv2.imwrite(output_path, merged_mask_downsampled)
    print(f"\nDownsampled merged mask saved to: {output_path}")
    print(f"Downsampled size: {new_height} x {new_width} pixels\n")
    
    # Free memory after processing each strategy
    del merged_mask, merged_mask_downsampled, contours
    gc.collect()

# Clean up large accumulators
# del non_zero_count, mip_accumulator, mean_accumulator, median_accumulator  # non_zero_count commented out
del mip_accumulator, mean_accumulator, median_accumulator
gc.collect()

# Print summary table
print("="*80)
print("SUMMARY OF ALL STRATEGIES")
print("="*80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("\n" + "="*80)

# Save summary to CSV
summary_path = os.path.join(output_dir, "stacking_strategies_summary.csv")
results_df.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")