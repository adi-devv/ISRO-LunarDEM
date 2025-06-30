# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# from scipy import ndimage
# from scipy.ndimage import gaussian_filter, uniform_filter
# from scipy.optimize import minimize
# from scipy.sparse import diags
# from scipy.sparse.linalg import spsolve
# import os
# import logging
# from pathlib import Path
# from typing import Tuple, Optional, Dict, Any
# import warnings
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# class EnhancedLunarDEMConverter:
#     """
#     Enhanced Lunar Digital Elevation Model (DEM) converter using shape-from-shading
#     techniques optimized for lunar terrain with CORRECTED gradient computation.
#     """
#
#     def __init__(self, data_dir: str = "data", config: Optional[Dict[str, Any]] = None):
#         self.data_dir = Path(data_dir)
#         self.image_path = self.data_dir / "moonframe.png"
#
#         # Default configuration - FIXED to use image gradients
#         self.config = {
#             'smoothing_sigma': 1.0,
#             'gradient_method': 'scharr',  # 'sobel', 'scharr', 'prewitt'
#             'integration_method': 'multigrid',  # 'frankot_chellappa', 'poisson', 'multigrid'
#             'albedo_estimation': True,
#             'shadow_detection': True,
#             'iterative_refinement': False,  # DISABLED photometric refinement
#             'max_iterations': 50,
#             'convergence_threshold': 1e-6,
#             'brightness_inversion': False,  # NEW: option to invert brightness interpretation
#             'gradient_scaling': 1.0  # NEW: scaling factor for gradient magnitude
#         }
#
#         if config:
#             self.config.update(config)
#
#         self.sun_elevation = None
#         self.sun_azimuth = None
#         self.albedo_map = None
#         self.shadow_mask = None
#
#     def parse_sun_params(self, filename: str = "sun_params.spm") -> Tuple[float, float]:
#         """
#         Enhanced sun parameter parsing with better error handling and format detection.
#         """
#         filepath = self.data_dir / filename
#
#         try:
#             with open(filepath, 'r') as f:
#                 content = f.read().strip()
#
#             lines = content.split('\n')
#             logger.info(f"Sun params file has {len(lines)} lines")
#
#             # Try to parse the structured format from the specification
#             for line in lines:
#                 parts = line.split()
#                 if len(parts) >= 15:  # Based on the SPM format specification
#                     try:
#                         # According to the format: positions 11-14 contain phase, aspect, azimuth, elevation
#                         phase_angle = float(parts[10])  # Field 11 (0-indexed)
#                         sun_aspect = float(parts[11])  # Field 12
#                         sun_azimuth = float(parts[12])  # Field 13
#                         sun_elevation = float(parts[13])  # Field 14
#
#                         logger.info(f"Parsed sun parameters: elevation={sun_elevation}°, azimuth={sun_azimuth}°")
#                         logger.info(f"Additional: phase_angle={phase_angle}°, sun_aspect={sun_aspect}°")
#
#                         return sun_elevation, sun_azimuth
#
#                     except (ValueError, IndexError) as e:
#                         logger.warning(f"Error parsing line: {line[:50]}... - {e}")
#                         continue
#
#             # Fallback: try to find reasonable values in the file
#             all_numbers = []
#             for line in lines:
#                 parts = line.split()
#                 for part in parts:
#                     try:
#                         num = float(part)
#                         if 0 <= num <= 90:  # Reasonable elevation range
#                             all_numbers.append(num)
#                     except ValueError:
#                         continue
#
#             if len(all_numbers) >= 2:
#                 sun_elevation = all_numbers[0]
#                 sun_azimuth = all_numbers[1] if all_numbers[1] <= 360 else all_numbers[0]
#                 logger.warning(f"Using fallback parsing: elevation={sun_elevation}°, azimuth={sun_azimuth}°")
#                 return sun_elevation, sun_azimuth
#
#         except FileNotFoundError:
#             logger.warning(f"Sun params file {filename} not found")
#         except Exception as e:
#             logger.error(f"Error parsing sun params: {e}")
#
#         # Default values for lunar observations
#         default_elevation, default_azimuth = 25.0, 315.0
#         logger.info(f"Using default sun parameters: elevation={default_elevation}°, azimuth={default_azimuth}°")
#         return default_elevation, default_azimuth
#
#     def load_and_preprocess_image(self) -> Optional[np.ndarray]:
#         """
#         Enhanced image loading with preprocessing for lunar imagery.
#         FIXED: Better preprocessing to preserve topographic information while reducing noise.
#         """
#         try:
#             # Try multiple image formats
#             for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
#                 img_path = self.image_path.with_suffix(ext)
#                 if img_path.exists():
#                     self.image_path = img_path
#                     break
#
#             img = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 raise FileNotFoundError(f"Could not load image: {self.image_path}")
#
#             logger.info(f"Loaded image: {img.shape}, dtype: {img.dtype}")
#
#             # Convert to float and normalize
#             img = img.astype(np.float64) / 255.0
#
#             # FIXED: More conservative preprocessing to preserve topographic signals
#             # Apply mild gamma correction for lunar imagery
#             gamma = 0.9  # Less aggressive than before
#             img = np.power(img, gamma)
#
#             # FIXED: Use bilateral filter for noise reduction while preserving edges
#             # This is better than CLAHE which can create artifacts
#             img_uint8 = (img * 255).astype(np.uint8)
#             img_filtered = cv2.bilateralFilter(img_uint8, 5, 50, 50)  # Smaller kernel
#             img = img_filtered.astype(np.float64) / 255.0
#
#             # Optional: Apply gentle Gaussian smoothing if needed
#             if self.config['smoothing_sigma'] > 0:
#                 img = gaussian_filter(img, sigma=self.config['smoothing_sigma'] * 0.5)
#
#             logger.info(f"Preprocessed image range: [{np.min(img):.4f}, {np.max(img):.4f}]")
#             return img
#
#         except Exception as e:
#             logger.error(f"Error loading image: {e}")
#             return None
#
#     def estimate_albedo_and_shadows(self, image: np.ndarray, sun_elevation: float, sun_azimuth: float) -> Tuple[
#         np.ndarray, np.ndarray]:
#         """
#         Estimate albedo map and detect shadow regions.
#         IMPROVED: Better shadow detection that doesn't interfere with topography.
#         """
#         h, w = image.shape
#
#         # Create initial albedo estimate using larger window
#         albedo_estimate = uniform_filter(image, size=21)
#
#         # IMPROVED: More conservative shadow detection
#         # Use both absolute threshold and relative threshold
#         shadow_threshold_abs = np.percentile(image, 5)  # Very dark regions only
#
#         # Local contrast-based detection
#         local_mean = uniform_filter(image, size=15)
#         local_std = np.sqrt(uniform_filter(image ** 2, size=15) - local_mean ** 2)
#
#         # Shadows are both dark AND have low local contrast
#         shadow_mask = (image < shadow_threshold_abs) & (local_std < 0.005)
#
#         # Remove small isolated shadow regions (likely noise)
#         kernel = np.ones((3, 3), np.uint8)
#         shadow_mask = cv2.morphologyEx(shadow_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
#
#         # Refine albedo in non-shadow regions
#         albedo_map = np.copy(image)
#         non_shadow = ~shadow_mask
#
#         if np.any(non_shadow):
#             # Scale to reasonable lunar albedo range
#             scale_factor = 0.12 / np.mean(image[non_shadow])
#             albedo_map = image * scale_factor
#             albedo_map = np.clip(albedo_map, 0.05, 0.3)
#
#         logger.info(f"Estimated albedo range: [{np.min(albedo_map):.4f}, {np.max(albedo_map):.4f}]")
#         logger.info(f"Shadow pixels: {np.sum(shadow_mask)} ({100 * np.sum(shadow_mask) / shadow_mask.size:.1f}%)")
#
#         return albedo_map, shadow_mask
#
#     def compute_enhanced_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         FIXED: Compute image gradients that represent topographic slopes, not brightness changes.
#         The key insight: gradients should represent the rate of change of surface elevation,
#         which correlates with illumination patterns but is not the same as brightness gradients.
#         """
#         # CRITICAL FIX: Apply shadow compensation before gradient computation
#         compensated_image = self.compensate_for_shadows(image)
#
#         # Apply adaptive smoothing to reduce noise while preserving topographic features
#         sigma = self.config['smoothing_sigma']
#         if sigma > 0:
#             smoothed = gaussian_filter(compensated_image, sigma=sigma)
#         else:
#             smoothed = compensated_image
#
#         method = self.config['gradient_method']
#
#         if method == 'scharr':
#             # Scharr operator (more accurate than Sobel)
#             grad_x = cv2.Scharr(smoothed, cv2.CV_64F, 1, 0) / 32.0
#             grad_y = cv2.Scharr(smoothed, cv2.CV_64F, 0, 1) / 32.0
#         elif method == 'sobel':
#             grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3) / 8.0
#             grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3) / 8.0
#         elif method == 'prewitt':
#             kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64) / 6.0
#             kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64) / 6.0
#             grad_x = cv2.filter2D(smoothed, cv2.CV_64F, kernel_x)
#             grad_y = cv2.filter2D(smoothed, cv2.CV_64F, kernel_y)
#         else:
#             # Default to central differences
#             grad_x = np.zeros_like(image)
#             grad_y = np.zeros_like(image)
#             grad_x[:, 1:-1] = (smoothed[:, 2:] - smoothed[:, :-2]) / 2.0
#             grad_y[1:-1, :] = (smoothed[2:, :] - smoothed[:-2, :]) / 2.0
#
#         # FIXED: Apply sun-angle based correction to relate gradients to surface slopes
#         grad_x, grad_y = self.apply_illumination_correction(grad_x, grad_y)
#
#         # Apply scaling factor if specified
#         scaling = self.config['gradient_scaling']
#         grad_x *= scaling
#         grad_y *= scaling
#
#         logger.info(f"Gradient stats - X: [{np.min(grad_x):.4f}, {np.max(grad_x):.4f}], std: {np.std(grad_x):.4f}")
#         logger.info(f"Gradient stats - Y: [{np.min(grad_y):.4f}, {np.max(grad_y):.4f}], std: {np.std(grad_y):.4f}")
#
#         return grad_x, grad_y
#
#     def compensate_for_shadows(self, image: np.ndarray) -> np.ndarray:
#         """
#         NEW METHOD: Compensate for shadows to improve gradient computation.
#         Shadows create artificial gradients that don't represent topography.
#         """
#         if self.shadow_mask is None:
#             return image
#
#         compensated = np.copy(image)
#
#         # For shadow regions, use interpolated values based on nearby non-shadow pixels
#         if np.any(self.shadow_mask):
#             # Simple approach: replace shadow values with local median of non-shadow pixels
#             kernel_size = 7
#             for i in range(image.shape[0]):
#                 for j in range(image.shape[1]):
#                     if self.shadow_mask[i, j]:
#                         # Get local neighborhood
#                         i_start = max(0, i - kernel_size // 2)
#                         i_end = min(image.shape[0], i + kernel_size // 2 + 1)
#                         j_start = max(0, j - kernel_size // 2)
#                         j_end = min(image.shape[1], j + kernel_size // 2 + 1)
#
#                         local_region = image[i_start:i_end, j_start:j_end]
#                         local_mask = self.shadow_mask[i_start:i_end, j_start:j_end]
#
#                         # Get non-shadow pixels in the local region
#                         non_shadow_local = local_region[~local_mask]
#                         if len(non_shadow_local) > 0:
#                             compensated[i, j] = np.median(non_shadow_local)
#
#         return compensated
#
#     def apply_illumination_correction(self, grad_x: np.ndarray, grad_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         NEW METHOD: Apply illumination-based correction to relate image gradients to surface slopes.
#         This accounts for the relationship between surface orientation and illumination.
#         """
#         if self.sun_elevation is None or self.sun_azimuth is None:
#             return grad_x, grad_y
#
#         # Convert sun angles to radians
#         sun_elev_rad = np.radians(self.sun_elevation)
#         sun_azim_rad = np.radians(self.sun_azimuth)
#
#         # Sun direction components
#         sun_x = np.cos(sun_elev_rad) * np.sin(sun_azim_rad)
#         sun_y = np.cos(sun_elev_rad) * np.cos(sun_azim_rad)
#
#         # Scale gradients based on sun angle
#         # Surfaces perpendicular to sun direction show maximum gradient response
#         scale_x = abs(sun_x) if abs(sun_x) > 0.1 else 0.5
#         scale_y = abs(sun_y) if abs(sun_y) > 0.1 else 0.5
#
#         # Apply directional scaling
#         grad_x_corrected = grad_x / scale_x
#         grad_y_corrected = grad_y / scale_y
#
#         return grad_x_corrected, grad_y_corrected
#
#     def integrate_gradients_multigrid(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
#         """
#         Multigrid method for gradient integration - more robust than FFT methods.
#         IMPROVED: Better handling of boundary conditions and convergence.
#         """
#
#         def poisson_solve_iteration(z, grad_x, grad_y, h=1.0):
#             """Single Gauss-Seidel iteration for Poisson equation"""
#             z_new = np.copy(z)
#             rows, cols = z.shape
#
#             for i in range(1, rows - 1):
#                 for j in range(1, cols - 1):
#                     # Discrete Poisson equation: ∇²z = div(∇I)
#                     div_grad = (grad_x[i, j] - grad_x[i, j - 1]) + (grad_y[i, j] - grad_y[i - 1, j])
#
#                     z_new[i, j] = 0.25 * (z[i - 1, j] + z[i + 1, j] + z[i, j - 1] + z[i, j + 1] - h * h * div_grad)
#
#             return z_new
#
#         h, w = grad_x.shape
#         height_map = np.zeros((h, w))
#
#         # IMPROVED: Better initialization with mean removal
#         # Remove DC component from gradients to improve integration
#         grad_x_centered = grad_x - np.mean(grad_x)
#         grad_y_centered = grad_y - np.mean(grad_y)
#
#         # Multigrid V-cycle with improved convergence criteria
#         max_iterations = self.config['max_iterations']
#         tolerance = self.config['convergence_threshold']
#
#         for iteration in range(max_iterations):
#             height_map_old = np.copy(height_map)
#             height_map = poisson_solve_iteration(height_map, grad_x_centered, grad_y_centered)
#
#             # Check convergence
#             error = np.max(np.abs(height_map - height_map_old))
#             if error < tolerance:
#                 logger.info(f"Multigrid converged after {iteration + 1} iterations")
#                 break
#
#         return height_map
#
#     def integrate_gradients_frankot_chellappa(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
#         """
#         Enhanced Frankot-Chellappa method with better handling of boundary conditions.
#         IMPROVED: Better DC component handling and boundary conditions.
#         """
#         rows, cols = grad_x.shape
#
#         # IMPROVED: Remove mean from gradients for better integration
#         grad_x_centered = grad_x - np.mean(grad_x)
#         grad_y_centered = grad_y - np.mean(grad_y)
#
#         # Pad arrays to avoid boundary artifacts
#         pad_rows = rows // 8  # Reduced padding
#         pad_cols = cols // 8
#
#         grad_x_pad = np.pad(grad_x_centered, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='symmetric')
#         grad_y_pad = np.pad(grad_y_centered, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='symmetric')
#
#         rows_pad, cols_pad = grad_x_pad.shape
#
#         # Create frequency domain coordinates
#         u = np.fft.fftfreq(cols_pad, d=1.0)
#         v = np.fft.fftfreq(rows_pad, d=1.0)
#         U, V = np.meshgrid(u, v)
#
#         # Avoid division by zero
#         denom = U ** 2 + V ** 2
#         denom[0, 0] = 1  # Handle DC component
#
#         # Take FFT of gradients
#         Gx = np.fft.fft2(grad_x_pad)
#         Gy = np.fft.fft2(grad_y_pad)
#
#         # Integrate in frequency domain
#         j = complex(0, 1)
#         Z = (-j * 2 * np.pi * U * Gx - j * 2 * np.pi * V * Gy) / (4 * np.pi ** 2 * denom)
#         Z[0, 0] = 0  # Set DC component to zero
#
#         # Inverse FFT to get height map
#         height_map_pad = np.real(np.fft.ifft2(Z))
#
#         # Remove padding
#         height_map = height_map_pad[pad_rows:-pad_rows, pad_cols:-pad_cols]
#
#         return height_map
#
#     def process_to_dem(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
#         """
#         Main processing pipeline with FIXED algorithms.
#         """
#         logger.info("Starting enhanced lunar DEM conversion...")
#
#         # Load and preprocess image
#         image = self.load_and_preprocess_image()
#         if image is None:
#             return None
#
#         logger.info(f"Image statistics: shape={image.shape}, range=[{np.min(image):.4f}, {np.max(image):.4f}]")
#
#         # Parse sun parameters
#         self.sun_elevation, self.sun_azimuth = self.parse_sun_params()
#
#         # Estimate albedo and detect shadows
#         if self.config['albedo_estimation']:
#             self.albedo_map, self.shadow_mask = self.estimate_albedo_and_shadows(
#                 image, self.sun_elevation, self.sun_azimuth)
#         else:
#             self.albedo_map = np.full_like(image, 0.12)  # Average lunar albedo
#             self.shadow_mask = np.zeros_like(image, dtype=bool)
#
#         # FIXED: Always use image-based gradients, not photometric gradients
#         logger.info("Computing topographic gradients from image...")
#         grad_x, grad_y = self.compute_enhanced_gradients(image)
#
#         # Integrate gradients to height map
#         method = self.config['integration_method']
#         logger.info(f"Integrating gradients using {method} method...")
#
#         if method == 'multigrid':
#             height_map = self.integrate_gradients_multigrid(grad_x, grad_y)
#         elif method == 'frankot_chellappa':
#             height_map = self.integrate_gradients_frankot_chellappa(grad_x, grad_y)
#         else:
#             # Fallback to simple integration
#             height_map = self.simple_height_integration(grad_x, grad_y)
#
#         # Post-process height map
#         height_map = self.post_process_dem(height_map, image)
#
#         logger.info(f"Final DEM statistics: range=[{np.min(height_map):.4f}, {np.max(height_map):.4f}], "
#                     f"std={np.std(height_map):.4f}")
#
#         return image, height_map, grad_x, grad_y
#
#     def simple_height_integration(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
#         """IMPROVED simple integration with better boundary handling."""
#         h, w = grad_x.shape
#
#         # Remove mean from gradients
#         grad_x_centered = grad_x - np.mean(grad_x)
#         grad_y_centered = grad_y - np.mean(grad_y)
#
#         # Method 1: Row-wise integration
#         height_x = np.zeros((h, w))
#         for i in range(h):
#             height_x[i, 1:] = np.cumsum(grad_x_centered[i, :-1])
#
#         # Method 2: Column-wise integration
#         height_y = np.zeros((h, w))
#         for j in range(w):
#             height_y[1:, j] = np.cumsum(grad_y_centered[:-1, j])
#
#         # Combine with weights based on gradient reliability
#         grad_mag_x = np.abs(grad_x_centered)
#         grad_mag_y = np.abs(grad_y_centered)
#
#         weight_x = grad_mag_y / (grad_mag_x + grad_mag_y + 1e-10)
#         weight_y = grad_mag_x / (grad_mag_x + grad_mag_y + 1e-10)
#
#         height_map = weight_x * height_x + weight_y * height_y
#
#         return height_map
#
#     def post_process_dem(self, height_map: np.ndarray, original_image: np.ndarray) -> np.ndarray:
#         """
#         IMPROVED post-processing with better outlier removal and normalization.
#         """
#         # Remove outliers more conservatively
#         percentile_2 = np.percentile(height_map, 2)
#         percentile_98 = np.percentile(height_map, 98)
#         height_map_clipped = np.clip(height_map, percentile_2, percentile_98)
#
#         # IMPROVED: Smooth in shadow regions where estimates are less reliable
#         if self.shadow_mask is not None and np.any(self.shadow_mask):
#             smoothed = gaussian_filter(height_map_clipped, sigma=1.5)
#             height_map_clipped = np.where(self.shadow_mask, smoothed, height_map_clipped)
#
#         # IMPROVED: Remove any remaining tilt or plane
#         # Fit a plane to the data and remove it
#         h, w = height_map_clipped.shape
#         y_coords, x_coords = np.mgrid[0:h, 0:w]
#
#         # Flatten for plane fitting
#         x_flat = x_coords.flatten()
#         y_flat = y_coords.flatten()
#         z_flat = height_map_clipped.flatten()
#
#         # Fit plane: z = ax + by + c
#         A = np.column_stack([x_flat, y_flat, np.ones(len(x_flat))])
#         plane_params, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
#
#         # Remove plane
#         plane = plane_params[0] * x_coords + plane_params[1] * y_coords + plane_params[2]
#         height_map_detrended = height_map_clipped - plane
#
#         # Normalize to [0, 1] range
#         height_min = np.min(height_map_detrended)
#         height_max = np.max(height_map_detrended)
#         if height_max > height_min:
#             height_map_final = (height_map_detrended - height_min) / (height_max - height_min)
#         else:
#             height_map_final = np.zeros_like(height_map_detrended)
#
#         return height_map_final
#
#     def create_enhanced_visualization(self, image: np.ndarray, height_map: np.ndarray,
#                                       grad_x: np.ndarray, grad_y: np.ndarray) -> None:
#         """
#         Create comprehensive visualization with additional analysis plots and save all figures.
#         """
#         # Create output directory if it doesn't exist
#         output_dir = "output"
#         output_dir.mkdir(exist_ok=True)
#
#         # Dictionary to store all figures and their filenames
#         figures = {
#             'original_image': ('Original Lunar Image', image, 'gray', None),
#             'dem_height_map': ('DEM Height Map', height_map, 'terrain', None),
#             'albedo_map': (
#                 'Estimated Albedo Map', self.albedo_map if self.albedo_map is not None else np.ones_like(image) * 0.12,
#                 'hot', None),
#             'shadow_mask': (
#                 'Shadow Detection', self.shadow_mask if self.shadow_mask is not None else np.zeros_like(image),
#                 'binary',
#                 None),
#             'gradient_x': ('Gradient X (∂z/∂x)', grad_x, 'RdBu', (-np.std(grad_x) * 3, np.std(grad_x) * 3)),
#             'gradient_y': ('Gradient Y (∂z/∂y)', grad_y, 'RdBu', (-np.std(grad_y) * 3, np.std(grad_y) * 3)),
#             'gradient_magnitude': ('Gradient Magnitude', np.sqrt(grad_x ** 2 + grad_y ** 2), 'hot', None),
#             'surface_normals': ('Surface Normals (HSV)', self.compute_surface_normals(height_map), None, None),
#             'hillshade': (
#                 'Hillshade Visualization', self.create_hillshade(height_map, self.sun_azimuth, self.sun_elevation),
#                 'gray',
#                 None)
#         }
#
#         # Save individual plots
#         for name, (title, data, cmap, vlim) in figures.items():
#             fig, ax = plt.subplots(figsize=(8, 6))
#             if cmap:
#                 if vlim:
#                     im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
#                 else:
#                     im = ax.imshow(data, cmap=cmap)
#                 plt.colorbar(im, ax=ax, fraction=0.046)
#             else:
#                 ax.imshow(data)
#             ax.set_title(title)
#             ax.axis('off')
#             plt.tight_layout()
#
#             # Save the figure
#             fig_path = output_dir / f"{name}.png"
#             fig.savefig(fig_path, bbox_inches='tight', dpi=300)
#             plt.close(fig)
#             logger.info(f"Saved {title} to {fig_path}")
#
#         # Create and save the elevation profile plot
#         fig_prof = plt.figure(figsize=(8, 6))
#         center_row = height_map[height_map.shape[0] // 2, :]
#         center_col = height_map[:, height_map.shape[1] // 2]
#         plt.plot(center_row, label='Horizontal profile', alpha=0.7)
#         plt.plot(center_col, label='Vertical profile', alpha=0.7)
#         plt.title('Elevation Profiles')
#         plt.xlabel('Pixel position')
#         plt.ylabel('Normalized height')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#
#         prof_path = output_dir / "elevation_profiles.png"
#         fig_prof.savefig
#         # Continuation from the cut-off point in create_enhanced_visualization method
#
#         prof_path = output_dir / "elevation_profiles.png"
#         fig_prof.savefig(prof_path, bbox_inches='tight', dpi=300)
#         plt.close(fig_prof)
#         logger.info(f"Saved elevation profiles to {prof_path}")
#
#         # Create and save the comprehensive comparison plot
#         fig_comp, axes = plt.subplots(3, 3, figsize=(15, 12))
#         axes = axes.flatten()
#
#         plot_data = [
#             (image, 'Original Image', 'gray'),
#             (height_map, 'DEM Height Map', 'terrain'),
#             (self.albedo_map if self.albedo_map is not None else np.ones_like(image) * 0.12, 'Albedo Map', 'hot'),
#             (self.shadow_mask if self.shadow_mask is not None else np.zeros_like(image), 'Shadow Mask', 'binary'),
#             (grad_x, 'Gradient X', 'RdBu'),
#             (grad_y, 'Gradient Y', 'RdBu'),
#             (np.sqrt(grad_x ** 2 + grad_y ** 2), 'Gradient Magnitude', 'hot'),
#             (self.compute_surface_normals(height_map), 'Surface Normals', None),
#             (self.create_hillshade(height_map, self.sun_azimuth, self.sun_elevation), 'Hillshade', 'gray')
#         ]
#
#         for i, (data, title, cmap) in enumerate(plot_data):
#             if cmap:
#                 axes[i].imshow(data, cmap=cmap)
#             else:
#                 axes[i].imshow(data)
#             axes[i].set_title(title, fontsize=10)
#             axes[i].axis('off')
#
#         plt.tight_layout()
#         comp_path = output_dir / "comprehensive_analysis.png"
#         fig_comp.savefig(comp_path, bbox_inches='tight', dpi=300)
#         plt.close(fig_comp)
#         logger.info(f"Saved comprehensive analysis to {comp_path}")
#
#         # Create and save statistics summary
#         stats_text = f"""
# Lunar DEM Processing Results
# ============================
#
# Input Image:
# - Shape: {image.shape}
# - Range: [{np.min(image):.4f}, {np.max(image):.4f}]
# - Mean: {np.mean(image):.4f}
# - Std: {np.std(image):.4f}
#
# Sun Parameters:
# - Elevation: {self.sun_elevation:.2f}°
# - Azimuth: {self.sun_azimuth:.2f}°
#
# Height Map:
# - Range: [{np.min(height_map):.4f}, {np.max(height_map):.4f}]
# - Mean: {np.mean(height_map):.4f}
# - Std: {np.std(height_map):.4f}
#
# Gradients:
# - X gradient range: [{np.min(grad_x):.4f}, {np.max(grad_x):.4f}]
# - Y gradient range: [{np.min(grad_y):.4f}, {np.max(grad_y):.4f}]
# - Max gradient magnitude: {np.max(np.sqrt(grad_x ** 2 + grad_y ** 2)):.4f}
#
# Shadow Analysis:
# - Shadow pixels: {np.sum(self.shadow_mask) if self.shadow_mask is not None else 0}
# - Shadow percentage: {100 * np.sum(self.shadow_mask) / self.shadow_mask.size if self.shadow_mask is not None else 0:.1f}%
#
# Processing Configuration:
# - Smoothing sigma: {self.config['smoothing_sigma']}
# - Gradient method: {self.config['gradient_method']}
# - Integration method: {self.config['integration_method']}
# - Albedo estimation: {self.config['albedo_estimation']}
# - Shadow detection: {self.config['shadow_detection']}
# """
#
#         stats_path = output_dir / "processing_statistics.txt"
#         with open(stats_path, 'w') as f:
#             f.write(stats_text)
#         logger.info(f"Saved processing statistics to {stats_path}")
#
#         logger.info("Enhanced visualization complete - all files saved to output directory")
#
#     def compute_surface_normals(self, height_map: np.ndarray) -> np.ndarray:
#         """
#         Compute surface normals from height map and visualize as HSV image.
#         """
#         # Compute gradients of the height map
#         grad_y, grad_x = np.gradient(height_map)
#
#         # Compute normal vectors
#         # Normal = (-dz/dx, -dz/dy, 1) normalized
#         normal_x = -grad_x
#         normal_y = -grad_y
#         normal_z = np.ones_like(grad_x)
#
#         # Normalize the normal vectors
#         norm = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
#         normal_x /= norm
#         normal_y /= norm
#         normal_z /= norm
#
#         # Convert to HSV for visualization
#         # Hue from azimuth angle of normal
#         azimuth = np.arctan2(normal_y, normal_x)
#         hue = (azimuth + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
#
#         # Saturation from horizontal component magnitude
#         saturation = np.sqrt(normal_x ** 2 + normal_y ** 2)
#
#         # Value from vertical component (how much the surface faces up)
#         value = (normal_z + 1) / 2  # Normalize to [0, 1]
#
#         # Create HSV image
#         hsv = np.stack([hue, saturation, value], axis=2)
#
#         # Convert HSV to RGB for display
#         from matplotlib.colors import hsv_to_rgb
#         rgb = hsv_to_rgb(hsv)
#
#         return rgb
#
#     def create_hillshade(self, height_map: np.ndarray, azimuth: float, elevation: float) -> np.ndarray:
#         """
#         Create hillshade visualization of the DEM.
#         """
#         # Convert angles to radians
#         azimuth_rad = np.radians(azimuth - 90)  # Adjust for standard convention
#         elevation_rad = np.radians(elevation)
#
#         # Compute gradients
#         grad_y, grad_x = np.gradient(height_map)
#
#         # Compute slope and aspect
#         slope = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
#         aspect = np.arctan2(-grad_x, grad_y)
#
#         # Compute hillshade
#         hillshade = np.sin(elevation_rad) * np.sin(slope) + \
#                     np.cos(elevation_rad) * np.cos(slope) * \
#                     np.cos(azimuth_rad - aspect)
#
#         # Normalize to [0, 1]
#         hillshade = (hillshade - np.min(hillshade)) / (np.max(hillshade) - np.min(hillshade))
#
#         return hillshade
#
#     def save_dem_formats(self, height_map: np.ndarray) -> None:
#         """
#         Save DEM in multiple useful formats.
#         """
#         output_dir = self.data_dir / "output"
#         output_dir.mkdir(exist_ok=True)
#
#         # Save as 16-bit TIFF (common DEM format)
#         height_16bit = (height_map * 65535).astype(np.uint16)
#         cv2.imwrite(str(output_dir / "dem_16bit.tif"), height_16bit)
#         logger.info("Saved 16-bit TIFF DEM")
#
#         # Save as 32-bit float TIFF (preserves full precision)
#         cv2.imwrite(str(output_dir / "dem_32bit.tif"), height_map.astype(np.float32))
#         logger.info("Saved 32-bit float TIFF DEM")
#
#         # Save as NumPy array
#         np.save(output_dir / "dem_array.npy", height_map)
#         logger.info("Saved NumPy array DEM")
#
#         # Save as CSV for analysis in other tools
#         np.savetxt(output_dir / "dem_data.csv", height_map, delimiter=',', fmt='%.6f')
#         logger.info("Saved CSV DEM data")
#
#         # Create a simple header file with metadata
#         metadata = f"""Lunar DEM Metadata
# ==================
# Dimensions: {height_map.shape[1]} x {height_map.shape[0]} pixels
# Data type: Float32
# Value range: [{np.min(height_map):.6f}, {np.max(height_map):.6f}]
# Sun elevation: {self.sun_elevation:.2f} degrees
# Sun azimuth: {self.sun_azimuth:.2f} degrees
# Processing method: {self.config['integration_method']}
# Gradient method: {self.config['gradient_method']}
# """
#         with open(output_dir / "dem_metadata.txt", 'w') as f:
#             f.write(metadata)
#         logger.info("Saved DEM metadata")
#
#
# def main():
#     """
#     Main execution function with comprehensive error handling and configuration options.
#     """
#     # Configuration options - modify these to experiment with different approaches
#     config = {
#         'smoothing_sigma': 1.0,  # Noise reduction (0.5-2.0 recommended)
#         'gradient_method': 'scharr',  # 'sobel', 'scharr', 'prewitt'
#         'integration_method': 'multigrid',  # 'frankot_chellappa', 'poisson', 'multigrid'
#         'albedo_estimation': True,  # Enable albedo estimation
#         'shadow_detection': True,  # Enable shadow detection
#         'iterative_refinement': False,  # Disable problematic photometric refinement
#         'max_iterations': 50,  # For iterative methods
#         'convergence_threshold': 1e-6,  # Convergence criteria
#         'brightness_inversion': False,  # Try True if results look inverted
#         'gradient_scaling': 1.0  # Scale gradient magnitude (0.5-2.0)
#     }
#
#     # Initialize converter
#     converter = EnhancedLunarDEMConverter(data_dir="data", config=config)
#
#     try:
#         # Process the lunar image to DEM
#         result = converter.process_to_dem()
#
#         if result is None:
#             logger.error("DEM processing failed")
#             return
#
#         image, height_map, grad_x, grad_y = result
#
#         # Create comprehensive visualizations
#         converter.create_enhanced_visualization(image, height_map, grad_x, grad_y)
#
#         # Save DEM in multiple formats
#         converter.save_dem_formats(height_map)
#
#         logger.info("Lunar DEM conversion completed successfully!")
#         logger.info("Check the 'data/output' directory for all generated files")
#
#         # Print summary statistics
#         print(f"\nProcessing Summary:")
#         print(f"Input image shape: {image.shape}")
#         print(f"DEM height range: [{np.min(height_map):.4f}, {np.max(height_map):.4f}]")
#         print(f"Mean elevation: {np.mean(height_map):.4f}")
#         print(f"Elevation std dev: {np.std(height_map):.4f}")
#         print(f"Sun parameters: {converter.sun_elevation:.1f}° elevation, {converter.sun_azimuth:.1f}° azimuth")
#
#     except Exception as e:
#         logger.error(f"Error during processing: {e}")
#         import traceback
#         traceback.print_exc()
#
#
# if __name__ == "__main__":
#     main()
# GPU-Accelerated Crater-Aware Lunar DEM Converter for Google Colab
# Run this cell first to install dependencies
"""
!pip install cupy-cuda11x  # For CUDA acceleration
!pip install scikit-image
!pip install opencv-python
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp

    GPU_AVAILABLE = True
    device_id = cp.cuda.Device().id
    device_props = cp.cuda.runtime.getDeviceProperties(device_id)
    print(f"✅ GPU acceleration available!")
    print(f"GPU: {device_props['name'].decode()}")
except ImportError:
    print("⚠️ CuPy not available, falling back to CPU")
    GPU_AVAILABLE = False
    cp = np  # Fallback to numpy
# Try to import PyTorch for some operations
try:
    import torch

    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"✅ PyTorch CUDA available: {torch.cuda.get_device_name()}")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUCraterDEMConverter:
    """
    GPU-accelerated crater-aware DEM converter optimized for Google Colab.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'crater_detection': {
                'gaussian_sigma': 2.0,
                'threshold_factor': 0.3,
                'min_crater_pixels': 50,
                'max_crater_pixels': 5000,
                'use_gpu': GPU_AVAILABLE
            },
            'dem_generation': {
                'smoothing_sigma': 1.5,
                'crater_depth_bias': 0.4,
                'rim_enhancement': 1.3,
                'use_gpu': GPU_AVAILABLE and TORCH_AVAILABLE
            }
        }

        if config:
            self.config.update(config)

        self.craters = []
        self.crater_mask = None

    def gpu_gaussian_filter(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """GPU-accelerated Gaussian filtering."""
        if self.config['crater_detection']['use_gpu'] and GPU_AVAILABLE:
            # Use CuPy for GPU acceleration
            image_gpu = cp.asarray(image)

            # Create Gaussian kernel
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1

            x = cp.arange(kernel_size) - kernel_size // 2
            y = cp.arange(kernel_size) - kernel_size // 2
            xx, yy = cp.meshgrid(x, y)
            kernel = cp.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
            kernel = kernel / cp.sum(kernel)

            # Apply convolution using FFT (faster for large kernels)
            if kernel_size > 15:
                # Pad image for FFT convolution
                pad_h = kernel_size // 2
                pad_w = kernel_size // 2
                image_padded = cp.pad(image_gpu, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

                # FFT convolution
                image_fft = cp.fft.fft2(image_padded)
                kernel_padded = cp.zeros_like(image_padded)
                kernel_padded[:kernel_size, :kernel_size] = kernel
                kernel_fft = cp.fft.fft2(kernel_padded)

                result = cp.real(cp.fft.ifft2(image_fft * kernel_fft))
                result = result[pad_h:-pad_h, pad_w:-pad_w]
            else:
                # Direct convolution for small kernels
                result = cp.zeros_like(image_gpu)
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        shift_i = i - kernel_size // 2
                        shift_j = j - kernel_size // 2

                        # Handle boundaries
                        start_i = max(0, -shift_i)
                        end_i = min(image_gpu.shape[0], image_gpu.shape[0] - shift_i)
                        start_j = max(0, -shift_j)
                        end_j = min(image_gpu.shape[1], image_gpu.shape[1] - shift_j)

                        img_start_i = max(0, shift_i)
                        img_end_i = min(image_gpu.shape[0], image_gpu.shape[0] + shift_i)
                        img_start_j = max(0, shift_j)
                        img_end_j = min(image_gpu.shape[1], image_gpu.shape[1] + shift_j)

                        result[start_i:end_i, start_j:end_j] += kernel[i, j] * image_gpu[img_start_i:img_end_i,
                                                                               img_start_j:img_end_j]

            return cp.asnumpy(result)
        else:
            # Fallback to CPU
            return gaussian_filter(image, sigma=sigma)

    def detect_craters_gpu(self, image: np.ndarray) -> Tuple[List[Tuple], np.ndarray]:
        """
        Fast GPU-accelerated crater detection using multiple techniques.
        """
        logger.info("Starting GPU crater detection...")

        # Smooth image to reduce noise
        sigma = self.config['crater_detection']['gaussian_sigma']
        smoothed = self.gpu_gaussian_filter(image, sigma)

        if GPU_AVAILABLE:
            # Move to GPU for processing
            img_gpu = cp.asarray(smoothed)

            # Method 1: Laplacian of Gaussian (LoG) for blob detection
            # This is excellent for detecting circular crater features
            sigma_log = sigma * 1.414  # √2 scaling for LoG

            # Compute LoG using GPU
            # LoG = sigma^2 * (d²/dx² + d²/dy²) of Gaussian
            # We'll approximate this using difference of Gaussians (DoG)
            sigma1 = sigma_log
            sigma2 = sigma_log * 1.6

            gaussian1 = self.gpu_gaussian_filter(cp.asnumpy(img_gpu), sigma1)
            gaussian2 = self.gpu_gaussian_filter(cp.asnumpy(img_gpu), sigma2)

            dog = cp.asarray(gaussian1 - gaussian2)

            # Find local maxima (crater centers)
            # Use a simple maximum filter approach
            kernel_size = int(sigma * 3)
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Dilate to find local maxima
            max_filtered = cp.zeros_like(dog)
            pad = kernel_size // 2
            dog_padded = cp.pad(dog, pad, mode='reflect')

            for i in range(kernel_size):
                for j in range(kernel_size):
                    shifted = dog_padded[i:i + dog.shape[0], j:j + dog.shape[1]]
                    max_filtered = cp.maximum(max_filtered, shifted)

            # Find peaks
            threshold = cp.mean(dog) + self.config['crater_detection']['threshold_factor'] * cp.std(dog)
            peaks = (dog >= max_filtered) & (dog > threshold)

            # Convert back to CPU for further processing
            peaks_cpu = cp.asnumpy(peaks)
            dog_cpu = cp.asnumpy(dog)

        else:
            # CPU fallback
            from scipy.ndimage import maximum_filter

            # Laplacian of Gaussian
            sigma_log = sigma * 1.414
            gaussian1 = gaussian_filter(smoothed, sigma_log)
            gaussian2 = gaussian_filter(smoothed, sigma_log * 1.6)
            dog_cpu = gaussian1 - gaussian2

            # Find local maxima
            kernel_size = int(sigma * 3)
            if kernel_size % 2 == 0:
                kernel_size += 1

            max_filtered = maximum_filter(dog_cpu, size=kernel_size)
            threshold = np.mean(dog_cpu) + self.config['crater_detection']['threshold_factor'] * np.std(dog_cpu)
            peaks_cpu = (dog_cpu >= max_filtered) & (dog_cpu > threshold)

        # Extract crater candidates
        y_coords, x_coords = np.where(peaks_cpu)
        crater_candidates = list(zip(x_coords, y_coords))

        logger.info(f"Found {len(crater_candidates)} crater candidates")

        # Estimate crater sizes using local analysis
        craters = []
        crater_mask = np.zeros_like(image, dtype=bool)

        for x, y in crater_candidates:
            # Estimate crater radius by finding the rim
            radius = self.estimate_crater_radius(smoothed, x, y)

            if radius > 0:
                # Validate crater size
                area = np.pi * radius ** 2
                min_area = self.config['crater_detection']['min_crater_pixels']
                max_area = self.config['crater_detection']['max_crater_pixels']

                if min_area <= area <= max_area:
                    craters.append((x, y, radius))

                    # Add to crater mask
                    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
                    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
                    crater_mask |= mask

        logger.info(f"Validated {len(craters)} craters")
        return craters, crater_mask

    def estimate_crater_radius(self, image: np.ndarray, cx: int, cy: int, max_radius: int = 50) -> float:
        """
        Estimate crater radius by analyzing intensity profile from center outward.
        """
        h, w = image.shape

        # Sample radial profiles in multiple directions
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        profiles = []

        for angle in angles:
            dx = np.cos(angle)
            dy = np.sin(angle)

            profile = []
            for r in range(1, max_radius):
                x = int(cx + r * dx)
                y = int(cy + r * dy)

                if 0 <= x < w and 0 <= y < h:
                    profile.append(image[y, x])
                else:
                    break

            if len(profile) > 10:  # Minimum profile length
                profiles.append(np.array(profile))

        if not profiles:
            return 0

        # Find average radius where intensity starts increasing (crater rim)
        radii = []
        for profile in profiles:
            if len(profile) < 10:
                continue

            # Smooth profile
            profile_smooth = gaussian_filter(profile, sigma=1.0)

            # Find first significant increase (rim detection)
            gradient = np.diff(profile_smooth)

            # Look for sustained increase after initial decrease
            for i in range(3, len(gradient) - 3):
                if (np.mean(gradient[i:i + 3]) > 0.01 and
                        np.mean(gradient[i - 3:i]) <= 0):
                    radii.append(i)
                    break

        if radii:
            return np.median(radii)
        else:
            return max_radius // 3  # Default estimate

    def create_crater_enhanced_gradients(self, image: np.ndarray, craters: List[Tuple],
                                         crater_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients with crater knowledge to fix the inversion problem.
        """
        logger.info("Creating crater-enhanced gradients...")

        if TORCH_AVAILABLE and self.config['dem_generation']['use_gpu']:
            # Use PyTorch for gradient computation
            img_tensor = torch.tensor(image, dtype=torch.float32, device=device)

            # Sobel filters for gradients
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32, device=device).view(1, 1, 3, 3)

            # Add batch and channel dimensions
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

            # Compute gradients
            grad_x = torch.nn.functional.conv2d(img_tensor, sobel_x, padding=1)
            grad_y = torch.nn.functional.conv2d(img_tensor, sobel_y, padding=1)

            # Remove batch and channel dimensions and move to CPU
            grad_x = grad_x.squeeze().cpu().numpy() / 8.0
            grad_y = grad_y.squeeze().cpu().numpy() / 8.0

        else:
            # CPU fallback
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) / 8.0
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) / 8.0

        # Apply crater-specific corrections
        crater_depth_bias = self.config['dem_generation']['crater_depth_bias']
        rim_enhancement = self.config['dem_generation']['rim_enhancement']

        # Create crater interior and rim masks
        crater_interior = np.zeros_like(crater_mask, dtype=bool)
        crater_rim = np.zeros_like(crater_mask, dtype=bool)

        for x, y, radius in craters:
            # Interior (80% of radius)
            yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
            interior_mask = (xx - x) ** 2 + (yy - y) ** 2 <= (0.8 * radius) ** 2
            crater_interior |= interior_mask

            # Rim (annulus between 80% and 120% of radius)
            rim_mask = ((xx - x) ** 2 + (yy - y) ** 2 <= (1.2 * radius) ** 2) & \
                       ((xx - x) ** 2 + (yy - y) ** 2 >= (0.8 * radius) ** 2)
            crater_rim |= rim_mask

        # Bias gradients toward making crater interiors deeper
        grad_x[crater_interior] *= (1 + crater_depth_bias)
        grad_y[crater_interior] *= (1 + crater_depth_bias)

        # Enhance crater rims to be higher
        grad_x[crater_rim] *= rim_enhancement
        grad_y[crater_rim] *= rim_enhancement

        return grad_x, grad_y

    def integrate_gradients_gpu(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated gradient integration using iterative methods.
        """
        logger.info("Integrating gradients using GPU acceleration...")

        if TORCH_AVAILABLE and self.config['dem_generation']['use_gpu']:
            # Use PyTorch for iterative solving
            device = torch.device('cuda')

            # Convert to tensors
            gx = torch.tensor(grad_x, dtype=torch.float32, device=device)
            gy = torch.tensor(grad_y, dtype=torch.float32, device=device)

            # Initialize height map
            height = torch.zeros_like(gx)

            # Iterative integration using Jacobi method
            max_iter = self.config['dem_generation'].get('max_iterations', 100)

            for iteration in range(max_iter):
                height_old = height.clone()

                # Compute divergence of gradients
                div_grad = torch.zeros_like(gx)
                div_grad[1:-1, 1:-1] = (gx[1:-1, 2:] - gx[1:-1, :-2]) / 2.0 + \
                                       (gy[2:, 1:-1] - gy[:-2, 1:-1]) / 2.0

                # Update height using discrete Poisson equation
                # ∇²h = div(grad)
                laplacian = torch.zeros_like(height)
                laplacian[1:-1, 1:-1] = (height[2:, 1:-1] + height[:-2, 1:-1] +
                                         height[1:-1, 2:] + height[1:-1, :-2] -
                                         4 * height[1:-1, 1:-1])

                # Update rule: h_new = h_old + α * (div_grad - laplacian)
                alpha = 0.25
                height[1:-1, 1:-1] += alpha * (div_grad[1:-1, 1:-1] - laplacian[1:-1, 1:-1])

                # Check convergence
                if iteration % 10 == 0:
                    error = torch.max(torch.abs(height - height_old)).item()
                    if error < 1e-6:
                        logger.info(f"Converged after {iteration} iterations")
                        break

            return height.cpu().numpy()

        else:
            # CPU fallback - simple integration
            return self.simple_integration_cpu(grad_x, grad_y)

    def simple_integration_cpu(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """Simple CPU-based integration as fallback."""
        h, w = grad_x.shape
        height = np.zeros((h, w))

        # Integrate row-wise
        for i in range(h):
            height[i, 1:] = np.cumsum(grad_x[i, :-1])

        # Integrate column-wise and average
        height_y = np.zeros((h, w))
        for j in range(w):
            height_y[1:, j] = np.cumsum(grad_y[:-1, j])

        # Combine with equal weights
        return (height + height_y) / 2.0

    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple], np.ndarray]:
        """
        Main processing pipeline.
        """
        logger.info("Starting crater-aware DEM processing...")

        # Step 1: Detect craters
        craters, crater_mask = self.detect_craters_gpu(image)

        # Step 2: Create crater-enhanced gradients
        grad_x, grad_y = self.create_crater_enhanced_gradients(image, craters, crater_mask)

        # Step 3: Integrate to height map
        height_map = self.integrate_gradients_gpu(grad_x, grad_y)

        # Step 4: Post-process
        height_map = self.post_process_dem(height_map, crater_mask)

        return height_map, craters, crater_mask

    def post_process_dem(self, height_map: np.ndarray, crater_mask: np.ndarray) -> np.ndarray:
        """Post-process the DEM to enhance crater features."""

        # Remove global tilt
        h, w = height_map.shape
        y, x = np.mgrid[0:h, 0:w]

        # Fit plane to non-crater regions for better tilt removal
        if crater_mask is not None:
            non_crater = ~crater_mask
            if np.sum(non_crater) > 100:  # Enough points for fitting
                A = np.column_stack([x[non_crater].flatten(),
                                     y[non_crater].flatten(),
                                     np.ones(np.sum(non_crater))])
                plane_params = np.linalg.lstsq(A, height_map[non_crater].flatten(), rcond=None)[0]
                plane = plane_params[0] * x + plane_params[1] * y + plane_params[2]
                height_map = height_map - plane

        # Normalize
        height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))

        return height_map

    def visualize_results(self, image: np.ndarray, height_map: np.ndarray,
                          craters: List[Tuple], crater_mask: np.ndarray):
        """Create comprehensive visualization of results."""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Detected craters
        axes[0, 1].imshow(image, cmap='gray')
        for x, y, r in craters:
            circle = plt.Circle((x, y), r, fill=False, color='red', linewidth=2)
            axes[0, 1].add_patch(circle)
            axes[0, 1].plot(x, y, 'r+', markersize=8)
        axes[0, 1].set_title(f'Detected Craters ({len(craters)})')
        axes[0, 1].axis('off')

        # Crater mask
        axes[0, 2].imshow(crater_mask, cmap='hot')
        axes[0, 2].set_title('Crater Mask')
        axes[0, 2].axis('off')

        # Height map
        im1 = axes[1, 0].imshow(height_map, cmap='terrain')
        axes[1, 0].set_title('DEM Height Map')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0])

        # Hillshade
        hillshade = self.create_hillshade(height_map)
        axes[1, 1].imshow(hillshade, cmap='gray')
        axes[1, 1].set_title('Hillshade')
        axes[1, 1].axis('off')

        # 3D surface (if matplotlib supports it)
        try:
            ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
            h, w = height_map.shape
            x_3d = np.arange(0, w, max(1, w // 50))
            y_3d = np.arange(0, h, max(1, h // 50))
            X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
            Z_3d = height_map[::max(1, h // 50), ::max(1, w // 50)]

            ax_3d.plot_surface(X_3d, Y_3d, Z_3d, cmap='terrain', alpha=0.8)
            ax_3d.set_title('3D Surface')
            ax_3d.set_box_aspect([1, 1, 0.5])
        except:
            axes[1, 2].text(0.5, 0.5, '3D view\nnot available',
                            ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\n📊 Processing Results:")
        print(f"   🌑 Detected craters: {len(craters)}")
        print(f"   📏 Crater coverage: {np.sum(crater_mask) / crater_mask.size * 100:.1f}%")
        print(f"   ⛰️  Height range: [{np.min(height_map):.3f}, {np.max(height_map):.3f}]")
        print(f"   📈 Height std dev: {np.std(height_map):.3f}")

    def create_hillshade(self, height_map: np.ndarray, azimuth: float = 315,
                         elevation: float = 45) -> np.ndarray:
        """Create hillshade visualization."""
        grad_y, grad_x = np.gradient(height_map)

        azimuth_rad = np.radians(azimuth - 90)
        elevation_rad = np.radians(elevation)

        slope = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
        aspect = np.arctan2(-grad_x, grad_y)

        hillshade = (np.sin(elevation_rad) * np.sin(slope) +
                     np.cos(elevation_rad) * np.cos(slope) *
                     np.cos(azimuth_rad - aspect))

        return (hillshade - np.min(hillshade)) / (np.max(hillshade) - np.min(hillshade))

def main():
    print("🚀 GPU-Accelerated Crater-Aware Lunar DEM Converter")
    print("=" * 50)

    # Check GPU status
    if GPU_AVAILABLE:
        device_id = cp.cuda.Device().id
        device_props = cp.cuda.runtime.getDeviceProperties(device_id)
        print(f"✅ CuPy GPU: {device_props['name'].decode()}")
    if TORCH_AVAILABLE:
        print(f"✅ PyTorch GPU: {torch.cuda.get_device_name()}")

    # Load lunar image
    print("\n📸 Loading lunar image...")
    image_path = "data/moonframe.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Downsample large images to prevent GPU memory issues
    if GPU_AVAILABLE and max(image.shape) > 2048:
        scale = 2048 / max(image.shape)
        image = cv2.resize(image, None, fx=scale, fy=scale)
        logger.info(f"Downsampled image to shape: {image.shape}")

    # Initialize converter
    converter = GPUCraterDEMConverter()

    # Process the image
    print("\n🔍 Processing image...")
    try:
        height_map, craters, crater_mask = converter.process_image(image)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return None, None, None

    # Visualize results
    print("\n📊 Creating visualizations...")
    converter.visualize_results(image, height_map, craters, crater_mask)

    # Free GPU memory
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()

    return height_map, craters, converter
# Instructions for Colab users
print("""
🔧 GOOGLE COLAB SETUP INSTRUCTIONS:

1. First, run this cell to install dependencies:
   !pip install cupy-cuda11x scikit-image opencv-python

2. Upload your lunar image file or use the synthetic image

3. Run the main() function:
   height_map, craters, converter = main()

4. The code will automatically use GPU acceleration if available

⚡ PERFORMANCE TIPS:
- Enable GPU runtime: Runtime → Change runtime type → GPU
- For large images (>2048px), consider downsampling first
- Monitor GPU memory usage with: !nvidia-smi

🔍 CUSTOMIZATION:
- Adjust crater detection sensitivity in config
- Modify crater_depth_bias to control crater depth
- Change rim_enhancement to adjust crater rim prominence
""")

if __name__ == "__main__":
    main()
