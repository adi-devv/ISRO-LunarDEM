import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter  # Keep gaussian_filter as it's from ndimage
# Removed `gradient` from here as it caused the error
import os  # Import os for path joining

# Although you don't have standard SPICE kernels, spiceypy is still useful
# for quaternion operations and vector math if we use them manually.
# If not installed: pip install spiceypy
# import spiceypy as spice

# --- 1. Configuration and File Paths ---
DATA_DIR = 'data/'  # Directory containing all parameter files

# Corrected paths to reflect they are in the 'data/' directory
IMAGE_PATH = os.path.join(DATA_DIR,'moonframe.png')
OATH_PATH = os.path.join(DATA_DIR, 'oath')
OAT_PATH = os.path.join(DATA_DIR, 'params.oat')  # Corrected: params.oat
LBR_PATH = os.path.join(DATA_DIR, 'lbr')
SPM_PATH = os.path.join(DATA_DIR, 'sun_params.spm')

# --- 2. Load Moon Image ---
try:
    img = Image.open(IMAGE_PATH).convert('L')  # Convert to grayscale
    image_array = np.array(img, dtype=np.float32)  # Use float for calculations
    print(f"Loaded image: {IMAGE_PATH} with shape {image_array.shape}")
except FileNotFoundError:
    print(f"Error: Image file not found at {IMAGE_PATH}")
    exit()

# Normalize image intensity to [0, 1] for photometric calculations
image_normalized = image_array / 255.0


# --- 3. Custom File Parsers based on Provided Formats ---

def parse_utc_time(parts, start_idx):
    """Parses UTC time from 7I4 format (YYYYDDDDMMMMHHHHMMMMSSSSMMMM)"""
    # Assuming standard space-separated fields as per format description
    # YYYY MON DD HH MM SS SSS
    year = int(parts[start_idx])
    month = int(parts[start_idx + 1])
    day = int(parts[start_idx + 2])
    hour = int(parts[start_idx + 3])
    minute = int(parts[start_idx + 4])
    second = int(parts[start_idx + 5])
    millisecond = int(parts[start_idx + 6])

    # Construct a datetime object
    dt_object = datetime(year, month, day, hour, minute, second, millisecond * 1000)
    return dt_object


def parse_oath_header(filepath):
    """Parses the OATH header file."""
    header = {}
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            parts = first_line.split()

            # Based on the format description
            header['record_type'] = parts[0]
            # Joining parts 1-3 for project name as it's "CHANDRAYAAN-2 MISSION"
            header['project_name'] = " ".join(parts[1:4])
            header['block_length_bytes'] = int(parts[4])
            header['station_id'] = parts[5]

            # Start UTC time for First OAT block (7I4)
            header['start_utc'] = parse_utc_time(parts, 6)

            # End UTC time for Last OAT block (7I4)
            header['end_utc'] = parse_utc_time(parts, 13)

            header['num_oat_records'] = int(parts[20])  # Adjusted index based on example line
            header['record_length_oat'] = int(parts[21])  # Adjusted index
            header['attitude_source'] = int(parts[22])  # Adjusted index
            header['mission_phase'] = int(parts[23])  # Adjusted index (1-Earth, 3-Moon)

        print(
            f"OATH Header: Project='{header['project_name']}', Mission Phase='{header['mission_phase']}' (1=Earth, 3=Moon)")
        return header
    except FileNotFoundError:
        print(f"Error: OATH file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error parsing OATH file {filepath}: {e}")
        return None


def parse_oat_file(filepath):
    """Parses the OAT data file."""
    data_records = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('ORBTATTD'):
                    parts = line.split()
                    try:
                        record = {}
                        record['record_type'] = parts[0]
                        record['physical_record_num'] = int(parts[1])
                        record['block_length_bytes'] = int(parts[2])

                        record['utc_time'] = parse_utc_time(parts, 3)  # Start at index 3 for 7I4

                        # Indices need to be checked carefully based on the exact splitting behavior
                        # from the provided example `oat` content:
                        # ORBTATTD     1 6282023  10  30  23  58  21  26       161985.314143 ...
                        # parts[0] = ORBTATTD
                        # parts[1] = 1 (record number)
                        # parts[2] = 628 (block length)
                        # parts[3] to parts[9] = YYYY MM DD HH MM SS SSS
                        # parts[10] = Lunar Position X
                        # parts[11] = Lunar Position Y
                        # parts[12] = Lunar Position Z
                        # parts[13] = Satellite Position X
                        # parts[14] = Satellite Position Y
                        # parts[15] = Satellite Position Z
                        # parts[16] = Satellite velocity X-dot
                        # parts[17] = Satellite velocity Y-dot
                        # parts[18] = Satellite velocity Z-dot
                        # parts[19] to parts[22] = S/C Attitude Q1, Q2, Q3, Q4 (Inertial to Body)
                        # parts[23] to parts[26] = Transformation Quaternion for Earth Fixed IAU frame
                        # parts[27] to parts[30] = Transformation Quaternion for Lunar Fixed IAU frame
                        # parts[31] = Latitude of sub-satellite point
                        # parts[32] = Longitude of sub-satellite point
                        # parts[33] = Solar Azimuth
                        # parts[34] = Solar Elevation
                        # parts[35] = Latitude
                        # parts[36] = Longitude
                        # parts[37] = Satellite altitude (kms)
                        # parts[38] = Angle between +Roll and Velocity Vector
                        # parts[39] = Eclipse Status
                        # parts[40] = Emission Angle
                        # parts[41] = Sun Angle w.r.t -ve Yaw (Phase angle)
                        # parts[42] = Angle between +Yaw and Nadir
                        # parts[43] = Slant Range (Km)
                        # parts[44] = Orbit No
                        # parts[45] = Solar Zenith Angle
                        # parts[46] = Angle between Payload FoV axis and velocity vector
                        # parts[47] = X (yaw) angle
                        # parts[48] = Y (roll) angle
                        # parts[49] = Z (pitch) angle

                        record['lunar_pos_xyz_j2000_earth_kms'] = np.array(
                            [float(parts[10]), float(parts[11]), float(parts[12])])
                        record['satellite_pos_xyz_j2000_kms'] = np.array(
                            [float(parts[13]), float(parts[14]), float(parts[15])])
                        record['satellite_vel_xyz_kms_sec'] = np.array(
                            [float(parts[16]), float(parts[17]), float(parts[18])])

                        record['sc_attitude_q_inertial_to_body'] = np.array(
                            [float(parts[19]), float(parts[20]), float(parts[21]), float(parts[22])])
                        record['q_earth_fixed_iau'] = np.array(
                            [float(parts[23]), float(parts[24]), float(parts[25]), float(parts[26])])
                        record['q_lunar_fixed_iau'] = np.array(
                            [float(parts[27]), float(parts[28]), float(parts[29]), float(parts[30])])

                        record['sub_satellite_lat_deg'] = float(parts[31])
                        record['sub_satellite_lon_deg'] = float(parts[32])
                        record['solar_azimuth_deg'] = float(parts[33])
                        record['solar_elevation_deg'] = float(parts[34])
                        record['latitude_deg'] = float(parts[35])
                        record['longitude_deg'] = float(parts[36])
                        record['satellite_altitude_kms'] = float(parts[37])
                        record['roll_vel_angle_deg'] = float(parts[38])
                        record['eclipse_status'] = int(parts[39])
                        record['emission_angle_deg'] = float(parts[40])
                        record['sun_angle_neg_yaw_phase_deg'] = float(parts[41])
                        record['yaw_nadir_angle_deg'] = float(parts[42])
                        record['slant_range_km'] = float(parts[43])
                        record['orbit_no'] = int(parts[44])
                        record['solar_zenith_angle_deg'] = float(parts[45])
                        record['fov_vel_angle_deg'] = float(parts[46])
                        record['x_yaw_angle_deg'] = float(parts[47])
                        record['y_roll_angle_deg'] = float(parts[48])
                        record['z_pitch_angle_deg'] = float(parts[49])

                        data_records.append(record)
                    except (ValueError, IndexError) as e:
                        print(f"Skipping malformed OAT line: {line.strip()} (Error: {e})")
        print(f"Parsed {len(data_records)} records from OAT file.")
        return data_records
    except FileNotFoundError:
        print(f"Error: OAT file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error parsing OAT file {filepath}: {e}")
        return []


def parse_spm_file(filepath):
    """Parses the SPM (Sun Parameter) file."""
    data_records = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('ORBTATTD'):
                    parts = line.split()
                    try:
                        record = {}
                        record['record_type'] = parts[0]
                        record['physical_record_num'] = int(parts[1])
                        record['block_length_bytes'] = int(parts[2])

                        record['utc_time'] = parse_utc_time(parts, 3)

                        # Adjusted indices for SPM based on the provided format:
                        # Satellite position X, Y, Z (F20.6) - parts[10] to [12]
                        # Satellite velocity X, Y, Z (F12.6) - parts[13] to [15]
                        record['satellite_pos_x_kms'] = float(parts[10])
                        record['satellite_pos_y_kms'] = float(parts[11])
                        record['satellite_pos_z_kms'] = float(parts[12])
                        record['satellite_vel_x_kms_sec'] = float(parts[13])
                        record['satellite_vel_y_kms_sec'] = float(parts[14])
                        record['satellite_vel_z_kms_sec'] = float(parts[15])

                        record['phase_angle_deg'] = float(parts[16])
                        record['sun_aspect_deg'] = float(parts[17])
                        record['sun_azimuth_deg'] = float(parts[18])
                        record['sun_elevation_deg'] = float(parts[19])
                        record['orbit_limb_direction'] = int(parts[20])

                        data_records.append(record)
                    except (ValueError, IndexError) as e:
                        print(f"Skipping malformed SPM line: {line.strip()} (Error: {e})")
        print(f"Parsed {len(data_records)} records from SPM file.")
        return data_records
    except FileNotFoundError:
        print(f"Error: SPM file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error parsing SPM file {filepath}: {e}")
        return []


# --- 4. Camera Intrinsic Parameters (PLACEHOLDERS - YOU MUST REPLACE) ---
# These values are CRITICAL for accurate DEM generation.
# You need to find the calibration report or specifications for the Chandrayaan-2 camera
# (TMC, TMC-2, IIRS, or OHRC) that took the 'moonframe.png' image.

# Example typical values (these are NOT actual Chandrayaan-2 values)
# If moonframe.png is from TMC-2, search for "Chandrayaan-2 TMC-2 camera calibration"
camera_intrinsic = {
    'focal_length_pixels': 2000.0,  # Example: focal length in pixels
    'principal_point_x': image_array.shape[1] / 2,  # Center of image
    'principal_point_y': image_array.shape[0] / 2,  # Center of image
    'distortion_coeffs': np.array([0.0, 0.0, 0.0, 0.0])
    # k1, k2, p1, p2 (radial, tangential). Assume no distortion for now.
}
print(f"\nCamera Intrinsics (PLACEHOLDERS): {camera_intrinsic}")
print("WARNING: Replace these with actual values from Chandrayaan-2 camera calibration!")

# --- 5. Find the relevant OAT/SPM record for the image ---
# In a real scenario, you'd get the image acquisition time from the PNG's EXIF data
# or a sidecar metadata file that links to the OAT/SPM entry.
# For this example, let's just pick the first valid record from OAT and SPM.
# You might need to refine this to match the *actual* image time.

oath_header = parse_oath_header(OATH_PATH)
oat_data = parse_oat_file(OAT_PATH)
spm_data = parse_spm_file(SPM_PATH)

if not oat_data or not spm_data:
    print("Error: Could not parse enough data from OAT or SPM files. Exiting.")
    exit()

# For demonstration, use the first record (closest in time, assuming image taken around then)
# In reality, you'd find the OAT/SPM record whose UTC time is closest to the image's capture time.
relevant_oat_record = oat_data[0]
relevant_spm_record = spm_data[0]

print(f"\nUsing OAT record from: {relevant_oat_record['utc_time']}")
print(f"Using SPM record from: {relevant_spm_record['utc_time']}")

# --- 6. Calculate Illumination and Viewing Geometry ---

# Solar Incidence Angle = 90 - Sun Elevation Angle (from SPM file format description)
solar_incidence_angle_deg = 90.0 - relevant_spm_record['sun_elevation_deg']
print(f"Solar Incidence Angle: {solar_incidence_angle_deg:.2f} degrees")

# Sun Azimuth and Elevation from SPM
sun_azimuth_rad = np.deg2rad(relevant_spm_record['sun_azimuth_deg'])
sun_elevation_rad = np.deg2rad(relevant_spm_record['sun_elevation_deg'])

# Sun vector (L) in the local terrain frame (X-East, Y-North, Z-Up)
# This vector points *from* the surface *to* the sun.
sun_vector_local_terrain = np.array([
    np.cos(sun_elevation_rad) * np.sin(sun_azimuth_rad),  # X (East)
    np.cos(sun_elevation_rad) * np.cos(sun_azimuth_rad),  # Y (North)
    np.sin(sun_elevation_rad)  # Z (Up)
])
# Normalize it
sun_vector_local_terrain = sun_vector_local_terrain / np.linalg.norm(sun_vector_local_terrain)

# Viewing Vector (V):
# For a nadir-looking camera, the viewing vector points from the camera *to* the surface.
# In the local terrain frame (X-East, Y-North, Z-Up), the camera is "above" the surface.
# So, the viewing vector from the surface *to the camera* would be [0, 0, 1] (straight up).
# The viewing vector (V) from the *camera to the surface* is then [0, 0, -1].
view_vector_local_terrain = np.array([0, 0, -1])  # Camera looking straight down

print(f"Sun Vector (Local Terrain Frame): {sun_vector_local_terrain}")
print(f"View Vector (Local Terrain Frame): {view_vector_local_terrain}")


# --- 7. Photoclinometry (Shape-from-Shading) Core Logic ---

def lommel_seeliger_intensity(normal, L, V, albedo=0.1):
    """
    Calculates expected intensity for a given normal using a simplified Lommel-Seeliger model.
    normal, L, V are unit vectors. albedo is typically ~0.1 for lunar regolith.
    """

    # Ensure all vectors are unit vectors
    normal = normal / np.linalg.norm(normal)
    L = L / np.linalg.norm(L)
    V = V / np.linalg.norm(V)

    # Cosine of incidence angle (L.N)
    cos_i = np.dot(L, normal)

    # Cosine of emission angle (V.N) (V is from surface to viewer)
    # Since V is from camera to surface, we need -(V.N) to get surface to viewer.
    cos_e = -np.dot(V, normal)

    # Check for valid angles (light hitting surface, camera seeing surface)
    # If cos_i < 0, it's in shadow. If cos_e < 0, it's a back-facing slope for the camera.
    if cos_i < 0 or cos_e < 0:
        return 0.0

        # Simplified Lommel-Seeliger form
    if (cos_i + cos_e) == 0:
        return 0.0

    intensity = albedo * (cos_i / (cos_i + cos_e))
    return intensity


# The `reconstruct_depth_from_normals` function is a very basic, path-dependent integration.
# For real SfS, Horn's algorithm or Poisson solvers (e.g., from `scikit-image`) are used for better results.
# I'll keep this function for conceptual completeness, but note its limitations.
def reconstruct_depth_from_normals(normals_map, boundary_value=0.0):
    """
    Integrates a field of surface normals (dz/dx, dz/dy) to reconstruct a depth map.
    This is a simplified integration and prone to drift.
    """
    rows, cols, _ = normals_map.shape
    depth_map = np.zeros((rows, cols), dtype=np.float32)

    depth_map[0, 0] = boundary_value  # Set a boundary condition

    # This is a very basic, path-dependent integration. Not ideal for real applications.
    # A Poisson solver (e.g., from scikit-image.restoration) would be much better.
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                continue

            nx, ny, nz = normals_map[r, c]

            if nz < 1e-6:  # Prevent division by zero for nearly vertical normals
                dz_dx = 0.0
                dz_dy = 0.0
            else:
                dz_dx = -nx / nz
                dz_dy = -ny / nz

            # Simple average of two paths (horizontal and vertical)
            if r > 0 and c > 0:
                depth_map[r, c] = ((depth_map[r - 1, c] + dz_dy) + (depth_map[r, c - 1] + dz_dx)) / 2.0
            elif r > 0:
                depth_map[r, c] = depth_map[r - 1, c] + dz_dy
            elif c > 0:
                depth_map[r, c] = depth_map[r, c - 1] + dz_dx

    return depth_map


def photoclinometry_iterative_sfs(image_intensity, L, V, initial_dem, iterations=50, learning_rate=0.01):
    """
    Conceptual iterative Shape-from-Shading (SfS) algorithm.
    This is a simplification of a real SfS solver.
    """
    dem = initial_dem.copy()
    rows, cols = image_intensity.shape

    albedo = 0.1  # Simple albedo guess for Moon

    print(f"Starting {iterations} iterations of conceptual SfS...")

    for i in range(iterations):
        # 1. Compute current surface normals (N) from the DEM
        # Use numpy.gradient to compute gradients. It returns (gradient_y, gradient_x)
        dz_dy, dz_dx = np.gradient(dem)

        # Normals N = [-dz/dx, -dz/dy, 1] / sqrt( (dz/dx)^2 + (dz/dy)^2 + 1 )
        # Ensure normals are unit vectors. Z component points 'up' (away from Moon).

        # Create a map of normals
        normals_map = np.zeros((rows, cols, 3), dtype=np.float32)
        magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)

        normals_map[:, :, 0] = -dz_dx / magnitude  # -dz/dx component (along X)
        normals_map[:, :, 1] = -dz_dy / magnitude  # -dz/dy component (along Y)
        normals_map[:, :, 2] = 1.0 / magnitude  # Z component (up)

        # 2. Compute expected image intensity (I_predicted) based on current normals and reflectance model
        I_predicted = np.zeros_like(image_intensity)
        for r in range(rows):
            for c in range(cols):
                N_pixel = normals_map[r, c, :]
                I_predicted[r, c] = lommel_seeliger_intensity(N_pixel, L, V, albedo=albedo)

        # Scale predicted intensity to match original range (for visualization/comparison)
        if I_predicted.max() > I_predicted.min():
            I_predicted_scaled = (I_predicted - I_predicted.min()) / (I_predicted.max() - I_predicted.min() + 1e-9)
        else:
            I_predicted_scaled = np.zeros_like(I_predicted)

        # 3. Compute difference (error)
        error = image_intensity - I_predicted_scaled

        # 4. Update DEM based on error (gradient descent type update)
        dem += learning_rate * error

        # Optional: Apply smoothing to the DEM to prevent noise amplification
        dem = gaussian_filter(dem, sigma=0.5)

        if (i + 1) % 10 == 0 or i == iterations - 1:
            print(f"  Iteration {i + 1}/{iterations}, Max Error: {np.max(np.abs(error)):.4f}")

    print("Conceptual SfS iterations complete.")
    return dem


# --- 8. DEM Generation ---

# Initial DEM guess: a flat surface at a reference altitude.
# We will use the satellite altitude from OAT for a rough reference.
initial_altitude_km = relevant_oat_record['satellite_altitude_kms']  # This is altitude *above the Moon's surface*
initial_dem_guess_m = np.full(image_array.shape, initial_altitude_km * 1000.0, dtype=np.float32)

# Run the iterative SfS (conceptual)
# This will produce a DEM in the arbitrary units of the iterative process,
# which needs to be scaled and offset to absolute meters.
dem_relative_from_sfs = photoclinometry_iterative_sfs(
    image_normalized,
    sun_vector_local_terrain,
    view_vector_local_terrain,
    initial_dem_guess_m,  # Pass a base altitude guess
    iterations=100,  # More iterations for potentially better convergence
    learning_rate=100.0  # Adjust learning rate based on magnitude of DEM values and errors
)

# --- 9. Final Scaling and Georeferencing (Conceptual) ---
# The DEM from SfS is *relative* height variations.
# To make it an absolute DEM in meters and connect it to lunar coordinates:
# 1. Scale the relative DEM:
#    The range of values in `dem_relative_from_sfs` needs to be mapped to a realistic
#    height range for lunar terrain.
#    Example: If the image covers a 1km crater, the height range might be 1km.
#    This scaling factor is hard to determine without ground truth or stereo.
#    Let's use a heuristic for now, mapping the normalized relative DEM to a
#    plausible lunar terrain height range (e.g., 50 meters of variation).

# Normalize the SFS output to 0-1
dem_scaled_0_1 = (dem_relative_from_sfs - dem_relative_from_sfs.min()) / \
                 (dem_relative_from_sfs.max() - dem_relative_from_sfs.min() + 1e-9)  # Add epsilon to avoid div by zero

# Map to a realistic height range in meters. (Guess, based on crater sizes)
# For example, assume the relief within the image is up to 100 meters.
assumed_relief_range_m = 100.0  # This is a major assumption.
absolute_dem_m = dem_scaled_0_1 * assumed_relief_range_m

# Add a base altitude to make it absolute.
# The `satellite_altitude_kms` in OAT is already relative to the Moon's surface.
# So, we can just use that as a reference.
# The 'latitude_deg', 'longitude_deg' in OAT might represent the central point.
base_altitude_at_center_m = relevant_oat_record['satellite_altitude_kms'] * 1000.0 - (assumed_relief_range_m / 2)
absolute_dem_m += base_altitude_at_center_m

print(f"\nGenerated Absolute DEM (Conceptual): Min {absolute_dem_m.min():.2f} m, Max {absolute_dem_m.max():.2f} m")
print("Note: Absolute scaling is a strong estimate without further ground truth.")

# --- 10. Visualization Capabilities ---
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Lunar Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(dem_relative_from_sfs, cmap='viridis')
plt.title('Relative Depth Map (from SfS)')
plt.colorbar(label='Relative Depth (Arbitrary Units)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(absolute_dem_m, cmap='terrain')  # 'terrain' or 'gist_earth' for elevation
plt.title('Absolute DEM (Conceptual, Meters)')
plt.colorbar(label='Elevation (meters)')
plt.axis('off')

plt.tight_layout()
plt.show()

# --- 11. Save DEM ---
# For a real DEM, you'd save as a GeoTIFF with coordinate system info.
# For now, save as a normalized PNG and raw NumPy array.
dem_normalized_for_png = ((absolute_dem_m - absolute_dem_m.min()) / \
                          (absolute_dem_m.max() - absolute_dem_m.min() + 1e-9) * 255).astype(np.uint8)
Image.fromarray(dem_normalized_for_png).save("generated_lunar_dem_absolute_conceptual.png")
print("Conceptual Absolute DEM saved as generated_lunar_dem_absolute_conceptual.png")

np.save("generated_lunar_dem_absolute_conceptual.npy", absolute_dem_m)
print("Raw Absolute DEM data saved as generated_lunar_dem_absolute_conceptual.npy")

print("\n--- Summary of Limitations & Next Steps ---")
print(
    "1. **Camera Intrinsic Parameters:** The most critical missing piece for accuracy. You *must* find the focal length, principal point, and distortion coefficients for the Chandrayaan-2 camera that took this image.")
print(
    "2. **Exact Image Time to OAT/SPM Record:** This code uses the first record. In reality, you'd match the image's timestamp precisely to an OAT/SPM record.")
print(
    "3. **Photometric Model:** The Lommel-Seeliger implementation is simplified. A robust SfS needs a more accurate and possibly iterative solver, potentially incorporating albedo variations.")
print(
    "4. **Absolute Scaling:** The `assumed_relief_range_m` is a guess. True absolute scaling requires ground truth (e.g., from LOLA DEM) or more advanced techniques.")
print(
    "5. **Georeferencing:** To make this a proper geospatial product (GeoTIFF), you'd need to project each pixel's (row, col) to (latitude, longitude) using the spacecraft's position, attitude, and camera model, then embed this info into the TIFF.")
print("6. **Error Propagation and Refinement:** Real SfS handles noise, shadows, and occlusions much more rigorously.")