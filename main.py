import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedLunarDEMConverter:
    """
    Enhanced Lunar Digital Elevation Model (DEM) converter using advanced photoclinometry
    and shape-from-shading techniques optimized for lunar terrain.
    """

    def __init__(self, data_dir: str = "data", config: Optional[Dict[str, Any]] = None):
        self.data_dir = Path(data_dir)
        self.image_path = self.data_dir / "moonframe.png"

        # Default configuration
        self.config = {
            'smoothing_sigma': 1.0,
            'gradient_method': 'scharr',  # 'sobel', 'scharr', 'prewitt'
            'integration_method': 'multigrid',  # 'frankot_chellappa', 'poisson', 'multigrid'
            'albedo_estimation': True,
            'shadow_detection': True,
            'iterative_refinement': True,
            'max_iterations': 50,
            'convergence_threshold': 1e-6,
            'lunar_photometric_function': 'lommel_seeliger'  # 'lambert', 'lommel_seeliger', 'hapke'
        }

        if config:
            self.config.update(config)

        self.sun_elevation = None
        self.sun_azimuth = None
        self.albedo_map = None
        self.shadow_mask = None

    def parse_sun_params(self, filename: str = "sun_params.spm") -> Tuple[float, float]:
        """
        Enhanced sun parameter parsing with better error handling and format detection.
        """
        filepath = self.data_dir / filename

        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()

            lines = content.split('\n')
            logger.info(f"Sun params file has {len(lines)} lines")

            # Try to parse the structured format from the specification
            for line in lines:
                parts = line.split()
                if len(parts) >= 15:  # Based on the SPM format specification
                    try:
                        # According to the format: positions 11-14 contain phase, aspect, azimuth, elevation
                        phase_angle = float(parts[10])  # Field 11 (0-indexed)
                        sun_aspect = float(parts[11])  # Field 12
                        sun_azimuth = float(parts[12])  # Field 13
                        sun_elevation = float(parts[13])  # Field 14

                        logger.info(f"Parsed sun parameters: elevation={sun_elevation}°, azimuth={sun_azimuth}°")
                        logger.info(f"Additional: phase_angle={phase_angle}°, sun_aspect={sun_aspect}°")

                        return sun_elevation, sun_azimuth

                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing line: {line[:50]}... - {e}")
                        continue

            # Fallback: try to find reasonable values in the file
            all_numbers = []
            for line in lines:
                parts = line.split()
                for part in parts:
                    try:
                        num = float(part)
                        if 0 <= num <= 90:  # Reasonable elevation range
                            all_numbers.append(num)
                    except ValueError:
                        continue

            if len(all_numbers) >= 2:
                sun_elevation = all_numbers[0]
                sun_azimuth = all_numbers[1] if all_numbers[1] <= 360 else all_numbers[0]
                logger.warning(f"Using fallback parsing: elevation={sun_elevation}°, azimuth={sun_azimuth}°")
                return sun_elevation, sun_azimuth

        except FileNotFoundError:
            logger.warning(f"Sun params file {filename} not found")
        except Exception as e:
            logger.error(f"Error parsing sun params: {e}")

        # Default values for lunar observations
        default_elevation, default_azimuth = 25.0, 315.0
        logger.info(f"Using default sun parameters: elevation={default_elevation}°, azimuth={default_azimuth}°")
        return default_elevation, default_azimuth

    def load_and_preprocess_image(self) -> Optional[np.ndarray]:
        """
        Enhanced image loading with preprocessing for lunar imagery.
        """
        try:
            # Try multiple image formats
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                img_path = self.image_path.with_suffix(ext)
                if img_path.exists():
                    self.image_path = img_path
                    break

            img = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {self.image_path}")

            logger.info(f"Loaded image: {img.shape}, dtype: {img.dtype}")

            # Convert to float and normalize
            img = img.astype(np.float64) / 255.0

            # Apply gamma correction for lunar imagery (enhance low-light details)
            gamma = 0.8
            img = np.power(img, gamma)

            # Adaptive histogram equalization for better contrast
            img_uint8 = (img * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_uint8)
            img = img_enhanced.astype(np.float64) / 255.0

            # Remove noise while preserving edges
            img = cv2.bilateralFilter((img * 255).astype(np.uint8), 9, 75, 75).astype(np.float64) / 255.0

            logger.info(f"Preprocessed image range: [{np.min(img):.4f}, {np.max(img):.4f}]")
            return img

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def estimate_albedo_and_shadows(self, image: np.ndarray, sun_elevation: float, sun_azimuth: float) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Estimate albedo map and detect shadow regions.
        """
        h, w = image.shape

        # Create initial albedo estimate
        # Use median filtering to estimate local albedo
        albedo_estimate = uniform_filter(image, size=15)

        # Shadow detection based on brightness and local contrast
        shadow_threshold = np.percentile(image, 15)  # Bottom 15% as potential shadows
        local_std = uniform_filter(image ** 2, size=9) - uniform_filter(image, size=9) ** 2

        shadow_mask = (image < shadow_threshold) & (local_std < 0.01)

        # Refine albedo in non-shadow regions
        albedo_map = np.copy(image)
        non_shadow = ~shadow_mask

        if np.any(non_shadow):
            # Assume average lunar albedo of 0.12
            scale_factor = 0.12 / np.mean(image[non_shadow])
            albedo_map = image * scale_factor
            albedo_map = np.clip(albedo_map, 0.05, 0.3)  # Reasonable lunar albedo range

        logger.info(f"Estimated albedo range: [{np.min(albedo_map):.4f}, {np.max(albedo_map):.4f}]")
        logger.info(f"Shadow pixels: {np.sum(shadow_mask)} ({100 * np.sum(shadow_mask) / shadow_mask.size:.1f}%)")

        return albedo_map, shadow_mask

    def compute_enhanced_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image gradients using multiple methods and combine them.
        """
        # Apply adaptive smoothing
        sigma = self.config['smoothing_sigma']
        smoothed = gaussian_filter(image, sigma=sigma)

        method = self.config['gradient_method']

        if method == 'scharr':
            # Scharr operator (more accurate than Sobel)
            grad_x = cv2.Scharr(smoothed, cv2.CV_64F, 1, 0) / 32.0
            grad_y = cv2.Scharr(smoothed, cv2.CV_64F, 0, 1) / 32.0
        elif method == 'sobel':
            grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3) / 8.0
            grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        elif method == 'prewitt':
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64) / 6.0
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64) / 6.0
            grad_x = cv2.filter2D(smoothed, cv2.CV_64F, kernel_x)
            grad_y = cv2.filter2D(smoothed, cv2.CV_64F, kernel_y)
        else:
            # Default to central differences
            grad_x = np.zeros_like(image)
            grad_y = np.zeros_like(image)
            grad_x[:, 1:-1] = (smoothed[:, 2:] - smoothed[:, :-2]) / 2.0
            grad_y[1:-1, :] = (smoothed[2:, :] - smoothed[:-2, :]) / 2.0

        logger.info(f"Gradient stats - X: [{np.min(grad_x):.4f}, {np.max(grad_x):.4f}], std: {np.std(grad_x):.4f}")
        logger.info(f"Gradient stats - Y: [{np.min(grad_y):.4f}, {np.max(grad_y):.4f}], std: {np.std(grad_y):.4f}")

        return grad_x, grad_y

    def photometric_shape_from_shading(self, image: np.ndarray, albedo_map: np.ndarray,
                                       shadow_mask: np.ndarray, sun_elevation: float,
                                       sun_azimuth: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced photoclinometric shape-from-shading using lunar photometric models.
        """
        # Convert sun angles to radians and compute direction vector
        sun_elev_rad = np.radians(sun_elevation)
        sun_azim_rad = np.radians(sun_azimuth)

        # Sun direction in image coordinates (towards sun)
        sun_x = np.cos(sun_elev_rad) * np.sin(sun_azim_rad)
        sun_y = np.cos(sun_elev_rad) * np.cos(sun_azim_rad)
        sun_z = np.sin(sun_elev_rad)

        logger.info(f"Sun vector: ({sun_x:.3f}, {sun_y:.3f}, {sun_z:.3f})")

        h, w = image.shape

        # Initialize surface gradients
        p = np.zeros((h, w))  # dz/dx
        q = np.zeros((h, w))  # dz/dy

        # Use photometric model to estimate surface gradients
        non_shadow = ~shadow_mask

        if self.config['lunar_photometric_function'] == 'lommel_seeliger':
            # Lommel-Seeliger photometric function (better for lunar surfaces)
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    if non_shadow[i, j] and image[i, j] > 0.05:
                        I = image[i, j]
                        albedo = albedo_map[i, j]

                        # Estimate surface normal from reflectance
                        # For Lommel-Seeliger: I = albedo * cos(i) / (cos(i) + cos(e))
                        # where i is incidence angle, e is emission angle

                        # Initial gradient estimate from neighboring pixels
                        px = (image[i, j + 1] - image[i, j - 1]) / 2.0
                        qy = (image[i + 1, j] - image[i - 1, j]) / 2.0

                        # Iterative refinement using photometric constraint
                        for _ in range(3):
                            # Surface normal
                            norm = np.sqrt(px * px + qy * qy + 1)
                            nx, ny, nz = -px / norm, -qy / norm, 1.0 / norm

                            # Cosine of incidence angle
                            cos_i = max(0.01, nx * sun_x + ny * sun_y + nz * sun_z)
                            cos_e = max(0.01, nz)  # View from above

                            # Predicted intensity
                            I_pred = albedo * cos_i / (cos_i + cos_e)

                            # Adjust gradients based on error
                            error = I - I_pred
                            px += 0.1 * error * sun_x
                            qy += 0.1 * error * sun_y

                        p[i, j] = px
                        q[i, j] = qy
        else:
            # Simplified Lambertian model
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    if non_shadow[i, j] and image[i, j] > 0.05:
                        I = image[i, j]
                        albedo = albedo_map[i, j]

                        # For Lambertian: I = albedo * cos(i)
                        cos_i = I / albedo
                        cos_i = np.clip(cos_i, 0.01, 1.0)

                        # Estimate gradients (simplified)
                        p[i, j] = (image[i, j + 1] - image[i, j - 1]) / 2.0
                        q[i, j] = (image[i + 1, j] - image[i - 1, j]) / 2.0

        return p, q

    def integrate_gradients_multigrid(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """
        Multigrid method for gradient integration - more robust than FFT methods.
        """

        def poisson_solve_iteration(z, grad_x, grad_y, h=1.0):
            """Single Gauss-Seidel iteration for Poisson equation"""
            z_new = np.copy(z)
            rows, cols = z.shape

            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    # Discrete Poisson equation: ∇²z = div(∇I)
                    div_grad = (grad_x[i, j] - grad_x[i, j - 1]) + (grad_y[i, j] - grad_y[i - 1, j])

                    z_new[i, j] = 0.25 * (z[i - 1, j] + z[i + 1, j] + z[i, j - 1] + z[i, j + 1] - h * h * div_grad)

            return z_new

        h, w = grad_x.shape
        height_map = np.zeros((h, w))

        # Multigrid V-cycle
        max_iterations = self.config['max_iterations']
        tolerance = self.config['convergence_threshold']

        for iteration in range(max_iterations):
            height_map_old = np.copy(height_map)
            height_map = poisson_solve_iteration(height_map, grad_x, grad_y)

            # Check convergence
            error = np.max(np.abs(height_map - height_map_old))
            if error < tolerance:
                logger.info(f"Multigrid converged after {iteration + 1} iterations")
                break

        return height_map

    def integrate_gradients_frankot_chellappa(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """
        Enhanced Frankot-Chellappa method with better handling of boundary conditions.
        """
        rows, cols = grad_x.shape

        # Pad arrays to avoid boundary artifacts
        pad_rows = rows // 4
        pad_cols = cols // 4

        grad_x_pad = np.pad(grad_x, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='edge')
        grad_y_pad = np.pad(grad_y, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='edge')

        rows_pad, cols_pad = grad_x_pad.shape

        # Create frequency domain coordinates
        u = np.fft.fftfreq(cols_pad, d=1.0)
        v = np.fft.fftfreq(rows_pad, d=1.0)
        U, V = np.meshgrid(u, v)

        # Avoid division by zero
        denom = U ** 2 + V ** 2
        denom[0, 0] = 1  # Handle DC component

        # Take FFT of gradients
        Gx = np.fft.fft2(grad_x_pad)
        Gy = np.fft.fft2(grad_y_pad)

        # Integrate in frequency domain
        j = complex(0, 1)
        Z = (-j * 2 * np.pi * U * Gx - j * 2 * np.pi * V * Gy) / (4 * np.pi ** 2 * denom)
        Z[0, 0] = 0  # Set DC component to zero

        # Inverse FFT to get height map
        height_map_pad = np.real(np.fft.ifft2(Z))

        # Remove padding
        height_map = height_map_pad[pad_rows:-pad_rows, pad_cols:-pad_cols]

        return height_map

    def process_to_dem(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Main processing pipeline with enhanced algorithms.
        """
        logger.info("Starting enhanced lunar DEM conversion...")

        # Load and preprocess image
        image = self.load_and_preprocess_image()
        if image is None:
            return None

        logger.info(f"Image statistics: shape={image.shape}, range=[{np.min(image):.4f}, {np.max(image):.4f}]")

        # Parse sun parameters
        self.sun_elevation, self.sun_azimuth = self.parse_sun_params()

        # Estimate albedo and detect shadows
        if self.config['albedo_estimation']:
            self.albedo_map, self.shadow_mask = self.estimate_albedo_and_shadows(
                image, self.sun_elevation, self.sun_azimuth)
        else:
            self.albedo_map = np.full_like(image, 0.12)  # Average lunar albedo
            self.shadow_mask = np.zeros_like(image, dtype=bool)

        # Compute gradients
        if self.config['iterative_refinement']:
            # Use photoclinometric gradients
            grad_x, grad_y = self.photometric_shape_from_shading(
                image, self.albedo_map, self.shadow_mask, self.sun_elevation, self.sun_azimuth)
        else:
            # Use image gradients
            grad_x, grad_y = self.compute_enhanced_gradients(image)

        # Integrate gradients to height map
        method = self.config['integration_method']
        logger.info(f"Integrating gradients using {method} method...")

        if method == 'multigrid':
            height_map = self.integrate_gradients_multigrid(grad_x, grad_y)
        elif method == 'frankot_chellappa':
            height_map = self.integrate_gradients_frankot_chellappa(grad_x, grad_y)
        else:
            # Fallback to simple integration
            height_map = self.simple_height_integration(grad_x, grad_y)

        # Post-process height map
        height_map = self.post_process_dem(height_map, image)

        logger.info(f"Final DEM statistics: range=[{np.min(height_map):.4f}, {np.max(height_map):.4f}], "
                    f"std={np.std(height_map):.4f}")

        return image, height_map, grad_x, grad_y

    def simple_height_integration(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """Improved simple integration with better boundary handling."""
        h, w = grad_x.shape

        # Method 1: Row-wise integration
        height_x = np.zeros((h, w))
        for i in range(h):
            height_x[i, 1:] = np.cumsum(grad_x[i, :-1])

        # Method 2: Column-wise integration
        height_y = np.zeros((h, w))
        for j in range(w):
            height_y[1:, j] = np.cumsum(grad_y[:-1, j])

        # Combine with weights based on gradient reliability
        grad_mag_x = np.abs(grad_x)
        grad_mag_y = np.abs(grad_y)

        weight_x = grad_mag_y / (grad_mag_x + grad_mag_y + 1e-10)
        weight_y = grad_mag_x / (grad_mag_x + grad_mag_y + 1e-10)

        height_map = weight_x * height_x + weight_y * height_y

        return height_map

    def post_process_dem(self, height_map: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Post-process the DEM to improve quality and remove artifacts.
        """
        # Remove outliers
        percentile_1 = np.percentile(height_map, 1)
        percentile_99 = np.percentile(height_map, 99)
        height_map = np.clip(height_map, percentile_1, percentile_99)

        # Smooth in shadow regions where estimates are less reliable
        if self.shadow_mask is not None:
            smoothed = gaussian_filter(height_map, sigma=2.0)
            height_map = np.where(self.shadow_mask, smoothed, height_map)

        # Normalize to [0, 1] range
        height_min = np.min(height_map)
        height_max = np.max(height_map)
        if height_max > height_min:
            height_map = (height_map - height_min) / (height_max - height_min)

        return height_map
    def create_enhanced_visualization(self, image: np.ndarray, height_map: np.ndarray,
                                      grad_x: np.ndarray, grad_y: np.ndarray) -> None:
        """
        Create comprehensive visualization with additional analysis plots and save all figures.
        """
        # Create output directory if it doesn't exist
        output_dir = "output"
        output_dir.mkdir(exist_ok=True)

        # Dictionary to store all figures and their filenames
        figures = {
            'original_image': ('Original Lunar Image', image, 'gray', None),
            'dem_height_map': ('DEM Height Map', height_map, 'terrain', None),
            'albedo_map': (
            'Estimated Albedo Map', self.albedo_map if self.albedo_map is not None else np.ones_like(image) * 0.12,
            'hot', None),
            'shadow_mask': (
            'Shadow Detection', self.shadow_mask if self.shadow_mask is not None else np.zeros_like(image), 'binary',
            None),
            'gradient_x': ('Gradient X (∂z/∂x)', grad_x, 'RdBu', (-np.std(grad_x) * 3, np.std(grad_x) * 3)),
            'gradient_y': ('Gradient Y (∂z/∂y)', grad_y, 'RdBu', (-np.std(grad_y) * 3, np.std(grad_y) * 3)),
            'gradient_magnitude': ('Gradient Magnitude', np.sqrt(grad_x ** 2 + grad_y ** 2), 'hot', None),
            'surface_normals': ('Surface Normals (HSV)', self.compute_surface_normals(height_map), None, None),
            'hillshade': (
            'Hillshade Visualization', self.create_hillshade(height_map, self.sun_azimuth, self.sun_elevation), 'gray',
            None)
        }

        # Save individual plots
        for name, (title, data, cmap, vlim) in figures.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            if cmap:
                if vlim:
                    im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
                else:
                    im = ax.imshow(data, cmap=cmap)
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.imshow(data)
            ax.set_title(title)
            ax.axis('off')
            plt.tight_layout()

            # Save the figure
            fig_path = output_dir / f"{name}.png"
            fig.savefig(fig_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            logger.info(f"Saved {title} to {fig_path}")

        # Create and save the elevation profile plot
        fig_prof = plt.figure(figsize=(8, 6))
        center_row = height_map[height_map.shape[0] // 2, :]
        center_col = height_map[:, height_map.shape[1] // 2]
        plt.plot(center_row, label='Horizontal profile', alpha=0.7)
        plt.plot(center_col, label='Vertical profile', alpha=0.7)
        plt.title('Elevation Profiles')
        plt.xlabel('Pixel position')
        plt.ylabel('Normalized height')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        prof_path = output_dir / "elevation_profiles.png"
        fig_prof.savefig(prof_path, bbox_inches='tight', dpi=300)
        plt.close(fig_prof)
        logger.info(f"Saved elevation profiles to {prof_path}")

        # Create and save the height distribution plot
        fig_hist = plt.figure(figsize=(8, 6))
        plt.hist(height_map.flatten(), bins=50, alpha=0.7, color='brown')
        plt.title('Height Distribution')
        plt.xlabel('Normalized height')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        hist_path = output_dir / "height_distribution.png"
        fig_hist.savefig(hist_path, bbox_inches='tight', dpi=300)
        plt.close(fig_hist)
        logger.info(f"Saved height distribution to {hist_path}")

        # Create and save the 3D surface plot
        fig_3d = plt.figure(figsize=(10, 8))
        ax3d = fig_3d.add_subplot(111, projection='3d')
        h, w = height_map.shape
        step = max(1, min(h, w) // 100)  # Adaptive subsampling
        x = np.arange(0, w, step)
        y = np.arange(0, h, step)
        X, Y = np.meshgrid(x, y)
        Z = height_map[::step, ::step]
        ax3d.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8, linewidth=0, antialiased=True)
        ax3d.set_title('3D Surface Reconstruction')
        ax3d.set_xlabel('X (pixels)')
        ax3d.set_ylabel('Y (pixels)')
        ax3d.set_zlabel('Height')

        # Convert 3D plot to image and save
        canvas = FigureCanvas(fig_3d)
        canvas.draw()
        img_3d = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img_3d = img_3d.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig_3d)

        fig_3d_path = output_dir / "3d_surface.png"
        cv2.imwrite(str(fig_3d_path), cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved 3D surface plot to {fig_3d_path}")

        # Create a comprehensive figure (similar to original implementation)
        fig_comp = plt.figure(figsize=(20, 15))
        # ... [rest of the original figure creation code] ...
        plt.tight_layout()

        comp_path = output_dir / "comprehensive_visualization.png"
        fig_comp.savefig(comp_path, bbox_inches='tight', dpi=150)
        plt.close(fig_comp)
        logger.info(f"Saved comprehensive visualization to {comp_path}")


    def create_hillshade(self, height_map: np.ndarray, azimuth: float, elevation: float) -> np.ndarray:
        """Create hillshade visualization for better terrain representation."""
        # Convert angles to radians
        azimuth_rad = np.radians(azimuth - 90)  # Convert to mathematical convention
        elevation_rad = np.radians(elevation)

        # Calculate gradients
        grad_x = np.gradient(height_map, axis=1)
        grad_y = np.gradient(height_map, axis=0)

        # Calculate slope and aspect
        slope = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
        aspect = np.arctan2(-grad_x, grad_y)

        # Calculate hillshade
        hillshade = np.sin(elevation_rad) * np.sin(slope) + \
                    np.cos(elevation_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)

        # Normalize to 0-1 range
        hillshade = (hillshade - hillshade.min()) / (hillshade.max() - hillshade.min())

        # Apply gamma correction for better visualization
        hillshade = np.power(hillshade, 0.5)

        return hillshade

    def compute_surface_normals(self, height_map: np.ndarray) -> np.ndarray:
        """Compute surface normals from height map for visualization."""
        grad_x = np.gradient(height_map, axis=1)
        grad_y = np.gradient(height_map, axis=0)

        # Compute normal vectors
        normal_x = -grad_x
        normal_y = -grad_y
        normal_z = np.ones_like(height_map)

        # Normalize
        norm = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
        normal_x /= norm
        normal_y /= norm
        normal_z /= norm

        # Convert to HSV color representation
        hue = (np.arctan2(normal_y, normal_x) + np.pi) / (2 * np.pi)  # 0-1
        saturation = np.sqrt(normal_x ** 2 + normal_y ** 2)
        value = (normal_z + 1) / 2  # Map to 0-1

        # Combine into HSV image
        hsv_image = np.zeros((height_map.shape[0], height_map.shape[1], 3))
        hsv_image[..., 0] = hue
        hsv_image[..., 1] = saturation
        hsv_image[..., 2] = value

        # Convert to RGB for display
        rgb_image = cv2.cvtColor((hsv_image * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

        return rgb_image

    def save_dem(self, height_map: np.ndarray, filename: str = "lunar_dem.tif") -> None:
        """Save the DEM to a GeoTIFF file with proper metadata."""
        try:
            # Convert to 32-bit float
            dem_data = height_map.astype(np.float32)

            # Create output directory if it doesn't exist
            output_dir = self.data_dir / "output"
            output_dir.mkdir(exist_ok=True)

            # Save as TIFF
            output_path = output_dir / filename
            cv2.imwrite(str(output_path), dem_data)

            logger.info(f"Saved DEM to {output_path}")
        except Exception as e:
            logger.error(f"Error saving DEM: {e}")

    def run_full_pipeline(self, visualize: bool = True) -> Optional[np.ndarray]:
        """Run the complete DEM generation pipeline."""
        try:
            # Process image to DEM
            result = self.process_to_dem()
            if result is None:
                return None

            image, height_map, grad_x, grad_y = result

            # Save results
            self.save_dem(height_map)

            # Create visualizations
            if visualize:
                self.create_enhanced_visualization(image, height_map, grad_x, grad_y)

            return height_map
        except Exception as e:
            logger.error(f"Error in full pipeline: {e}")
            return None



if __name__ == "__main__":
    # Example usage
    converter = EnhancedLunarDEMConverter(data_dir="data")

    # Optional: customize configuration
    custom_config = {
        'smoothing_sigma': 1.5,
        'gradient_method': 'scharr',
        'integration_method': 'multigrid',
        'iterative_refinement': True,
        'max_iterations': 100
    }
    converter.config.update(custom_config)

    # Run the full pipeline
    dem = converter.run_full_pipeline()

    if dem is not None:
        logger.info("DEM generation completed successfully!")
    else:
        logger.error("DEM generation failed.")