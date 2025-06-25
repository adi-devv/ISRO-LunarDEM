import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from matplotlib.colors import LinearSegmentedColormap


def parse_spm_line(line):
    try:
        parts = line.strip().split()
        azimuth = float(parts[17])  # Solar azimuth (deg)
        elevation = float(parts[18])  # Solar elevation (deg)
        return azimuth, elevation
    except Exception as e:
        print(f"Error parsing line: {line}\n{e}")
        return None, None


def get_avg_sun_angles(spm_path):
    az_list = []
    el_list = []
    with open(spm_path, 'r') as f:
        for line in f:
            az, el = parse_spm_line(line)
            if az is not None and el is not None:
                az_list.append(az)
                el_list.append(el)
    if not az_list:
        raise ValueError("No valid solar angles found in .spm file.")
    az_mean = sum(az_list) / len(az_list)
    el_mean = sum(el_list) / len(el_list)
    return az_mean, el_mean


def poisson_solver(p, q):
    h, w = p.shape
    fx = np.fft.fftfreq(w).reshape(1, -1)
    fy = np.fft.fftfreq(h).reshape(-1, 1)

    # Compute divergence of gradients
    div = np.zeros_like(p)
    div[:, :-1] += p[:, :-1]
    div[:, 1:] -= p[:, :-1]
    div[:-1, :] += q[:-1, :]
    div[1:, :] -= q[:-1, :]

    div_fft = fft2(div)

    # Avoid divide-by-zero at the origin
    denom = (2 * np.pi * 1j * fx) ** 2 + (2 * np.pi * 1j * fy) ** 2
    denom[0, 0] = 1  # To avoid division by 0

    height_fft = div_fft / denom
    height_fft[0, 0] = 0  # Mean height = 0

    height = np.real(ifft2(height_fft))
    return height


def get_vibgyor_colormap():
    return LinearSegmentedColormap.from_list("vibgyor", [
        (0.0, '#8B00FF'),  # Violet
        (0.16, '#4B0082'),  # Indigo
        (0.33, '#0000FF'),  # Blue
        (0.50, '#00FF00'),  # Green
        (0.66, '#FFFF00'),  # Yellow
        (0.83, '#FF7F00'),  # Orange
        (1.0, '#FF0000')  # Red
    ])


def generate_dem(image_path, sun_azimuth_deg, sun_elevation_deg, out_path='output_dem.png',
                 out_grayscale_path='grayscale_dem.png'):
    img = cv2.imread(image_path, 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    sun_azimuth_rad = np.radians(sun_azimuth_deg)
    sun_elevation_rad = np.radians(sun_elevation_deg)

    # Convert sun angles to light direction vector
    light_dir = np.array([
        np.cos(sun_elevation_rad) * np.sin(sun_azimuth_rad),
        np.cos(sun_elevation_rad) * np.cos(sun_azimuth_rad),
        np.sin(sun_elevation_rad)
    ])

    Z = img.astype(np.float32) / 255.0
    p = np.gradient(Z, axis=1)
    q = np.gradient(Z, axis=0)

    DEM = poisson_solver(p, q)
    DEM = -DEM

    # Compute surface normals from DEM
    grad_x, grad_y = np.gradient(DEM)
    normals = np.stack([-grad_x, -grad_y, np.ones_like(DEM)], axis=-1)
    norm_magnitude = np.sqrt(np.sum(normals ** 2, axis=-1, keepdims=True))
    normals = normals / np.where(norm_magnitude > 0, norm_magnitude, 1)

    # Compute shading (cosine of angle between normal and light direction)
    shading = np.sum(normals * light_dir.reshape(1, 1, 3), axis=-1)
    shading = np.clip(shading, 0, 1)  # Ensure shading is in [0, 1]

    # Normalize and clip DEM for processing
    vmin, vmax = np.percentile(DEM, [2, 98])
    DEM_clipped = np.clip(DEM, vmin, vmax)
    DEM_norm = (DEM_clipped - vmin) / (vmax - vmin)

    # Blend DEM_norm with shading for grayscale
    shadow_weight = 0.7  # Adjust to control shadow influence
    grayscale = shadow_weight * (1 - shading) + (1 - shadow_weight) * DEM_norm
    grayscale = np.clip(grayscale, 0, 1)


    # Save grayscale image
    DEM_grayscale = (grayscale * 255).astype(np.uint8)
    cv2.imwrite(out_grayscale_path, DEM_grayscale)
    print(f"Grayscale DEM with sunlight shading saved to {out_grayscale_path}")

    # Save colored image with VIBGYOR colormap
    cmap = get_vibgyor_colormap()
    DEM_colored = (cmap(DEM_norm)[:, :, :3] * 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(DEM_colored, cv2.COLOR_RGB2BGR))
    print(f"DEM with custom VIBGYOR colormap saved to {out_path}")

    # Show normalized image with VIBGYOR colormap
    plt.imshow(DEM_norm, cmap=cmap)
    plt.title("Approximate DEM (VIBGYOR)")
    plt.colorbar(label='Relative Elevation')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    spm_file = 'sun_params.spm'  # Path to your .spm file
    image_file = 'moonframe.png'  # Path to your mono image

    sun_az, sun_el = get_avg_sun_angles(spm_file)
    print(f"Using Sun Azimuth: {sun_az:.2f} deg, Elevation: {sun_el:.2f} deg")

    generate_dem(image_file, sun_az, sun_el)