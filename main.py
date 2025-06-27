import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def extract_sun_params(file_path):
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip() or not line.startswith("ORBTATTD"):
                    continue
                parts = line.strip().split()
                try:
                    sun_azimuth = float(parts[15])  # Column 16
                    sun_elevation = float(parts[18])  # Column 19
                    return sun_azimuth, sun_elevation
                except (IndexError, ValueError) as parse_error:
                    print(f"Skipping line due to parse error: {parse_error}")
                    continue
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found.")
    except Exception as e:
        print(f"Unexpected error reading sun parameters: {e}")

    raise ValueError("Sun parameters not found or could not be parsed from the file.")


def load_preprocess_image(img_path, threshold=0.7, kernel_size=3):
    """Load image, apply Gaussian blur, and invert excessively bright pixels based on neighbors."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    # Apply Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Create a kernel for neighborhood analysis
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    # Pad the image to handle borders
    padded_img = np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)),
                        mode='reflect')

    # Output image
    output = img.copy()

    # Iterate over each pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > threshold:  # Check if pixel is excessively bright
                # Extract neighborhood
                neighborhood = padded_img[i:i + kernel_size, j:j + kernel_size]
                mean_neighbor = np.mean(neighborhood)

                # Invert intensity based on its own value
                inverted_value = 1.0 - img[i, j]

                # If surrounding pixels are also bright (mean > threshold), darken the main pixel further
                if mean_neighbor > threshold:
                    adjustment = (mean_neighbor - threshold) * 0.5  # Scale factor to darken
                    inverted_value = max(0.0, inverted_value - adjustment)

                output[i, j] = inverted_value

    return output


def generate_dem(image, sun_azimuth_deg, sun_elevation_deg):
    """Generate a relative DEM from shading using sun angle."""
    sun_azimuth_rad = np.radians(sun_azimuth_deg)
    sun_elevation_rad = np.radians(sun_elevation_deg)

    # Lighting vector
    L = np.array([
        np.cos(sun_elevation_rad) * np.sin(sun_azimuth_rad),
        np.cos(sun_elevation_rad) * np.cos(sun_azimuth_rad),
        np.sin(sun_elevation_rad)
    ])

    # Compute gradients
    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Approximate surface normals
    normal = np.dstack((-Gx, -Gy, np.ones_like(image)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= (norm + 1e-8)

    # Dot product for shading
    shading = np.clip(np.sum(normal * L, axis=2), 0, 1)

    # Pixel-wise elevation difference estimation
    dem = gaussian_filter(image - shading, sigma=1)

    return dem


def save_dem(dem, output_dir='output'):
    """Save DEM as grayscale and high-contrast color images to the output folder."""
    os.makedirs(output_dir, exist_ok=True)

    # Normalize DEM to 0–255 for image writing
    dem_normalized = cv2.normalize(dem, None, 0, 255, cv2.NORM_MINMAX)
    dem_uint8 = dem_normalized.astype(np.uint8)

    # Save grayscale DEM
    grayscale_path = os.path.join(output_dir, 'grayscale_dem.png')
    cv2.imwrite(grayscale_path, dem_uint8)

    grayscale_path = os.path.join(output_dir, 'grayscale_dem_grain.png')
    cv2.imwrite(grayscale_path, (dem * 255).astype(np.uint8))

    # Define a strong elevation color gradient (dark green → lime → yellow → gold → orange → purple)
    colors = [
        (0.0, '#004d00'),  # very dark green (lowest)
        (0.2, '#66cc00'),  # lime
        (0.4, '#ffff00'),  # yellow
        (0.6, '#ff9900'),  # gold/orange
        (0.8, '#ff3300'),  # orange-red
        (1.0, '#800080')  # purple (highest)
    ]
    cmap = LinearSegmentedColormap.from_list("elevation_colormap", colors)

    # Save color DEM using matplotlib
    colormap_path = os.path.join(output_dir, 'colormap_dem.png')
    plt.imsave(colormap_path, dem, cmap=cmap)

    print(f"✅ Saved grayscale DEM: {grayscale_path}")
    print(f"🌄 Saved color elevation DEM: {colormap_path}")


def main():
    # File paths
    img_path = "data/moonframe.png"
    spm_path = "data/sun_params.spm"
    output_dir = "output_p0"

    # Step 1: Get sun direction
    sun_azimuth, sun_elevation = extract_sun_params(spm_path)
    print(f"☀️ Sun Azimuth: {sun_azimuth:.2f}°, Elevation: {sun_elevation:.2f}°")

    # Step 2: Load and preprocess image with selective bright pixel inversion
    image = load_preprocess_image(img_path, threshold=0.7, kernel_size=3)

    # Step 3: Generate DEM
    dem = generate_dem(image, sun_azimuth, sun_elevation)

    # Step 4: Save outputs
    save_dem(dem, output_dir)

    print("🎉 DEM generated and saved successfully.")


if __name__ == "__main__":
    main()
