import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter
import os

 DATA_DIR = 'data/'   
IMAGE_PATH = os.path.join(DATA_DIR, 'moonframe.png')

 OATH_PATH = os.path.join(DATA_DIR, 'params.oath')
OAT_PATH = os.path.join(DATA_DIR, 'params.oat')
LBR_PATH = os.path.join(DATA_DIR, 'params.lbr')
SPM_PATH = os.path.join(DATA_DIR, 'sun_params.spm')

# Output file paths
OUTPUT_DEM_PNG = 'generated_lunar_dem_absolute_conceptual.png'
OUTPUT_DEM_NPY = 'generated_lunar_dem_absolute_conceptual.npy'

# SfS Parameters
NUM_SFS_ITERATIONS = 100
CONVERGENCE_THRESHOLD = 1e-4  # For SfS stopping condition (if error change is very small)
SFS_LEARNING_RATE = 100.0  # Adjusted based on your provided code for better initial convergence

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

def parse_utc_time_from_parts(parts_list):
    """
    Parses UTC time from a list of 7 string parts (YYYY, MM, DD, HH, MM, SS, SSS).
    Handles potential leading/trailing spaces in individual parts.
    """
    if len(parts_list) != 7:
        raise ValueError(f"Expected 7 parts for UTC time, got {len(parts_list)}: {parts_list}")

    try:
        year = int(parts_list[0].strip())
        month = int(parts_list[1].strip())
        day = int(parts_list[2].strip())
        hour = int(parts_list[3].strip())
        minute = int(parts_list[4].strip())
        second = int(parts_list[5].strip())
        millisecond = int(parts_list[6].strip())

        # Construct a datetime object
        dt_object = datetime(year, month, day, hour, minute, second, millisecond * 1000)
        return dt_object
    except ValueError as e:
        raise ValueError(f"Error converting time part to int from {parts_list}: {e}")


def extract_time_parts_from_28char_string(s):
    """
    Extracts 7 time components (YYYY, MM, DD, HH, MM, SS, SSS) from a 28-character string,
    assuming '7I4' format with space padding based on the example (YYYY MM DD HH MM SS SSS).
    """
    if len(s) != 28:
        raise ValueError(f"Expected 28-character time string, got '{s}' (length {len(s)})")

    # Extract 7 blocks of 4 characters, then strip spaces
    year = s[0:4].strip()
    month = s[4:8].strip()
    day = s[8:12].strip()
    hour = s[12:16].strip()
    minute = s[16:20].strip()
    second = s[20:24].strip()
    millisecond = s[24:28].strip()

    return [year, month, day, hour, minute, second, millisecond]


def parse_oath_header(filepath):
    """Parses the OATH header file using fixed-width parsing."""
    header = {}
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()

            # --- Fixed-width parsing based on the provided format ---
            # All indices are 0-based
            header['record_type'] = first_line[0:12].strip()  # A12 (0 to 11)
            header['project_name'] = first_line[12:33].strip()  # A21 (12 to 32)
            header['block_length_bytes'] = int(first_line[33:39].strip())  # I6 (33 to 38)
            header['station_id'] = first_line[39:43].strip()  # A4 (39 to 42)

            # Start UTC time: 7I4 (28 bytes total)
            start_utc_raw_str = first_line[43:71]  # 43 to 43+28-1 = 70
            header['start_utc'] = parse_utc_time_from_parts(extract_time_parts_from_28char_string(start_utc_raw_str))

            # End UTC time: 7I4 (28 bytes total)
            end_utc_raw_str = first_line[71:99]  # 71 to 71+28-1 = 98
            header['end_utc'] = parse_utc_time_from_parts(extract_time_parts_from_28char_string(end_utc_raw_str))

            header['num_oat_records'] = int(first_line[99:105].strip())  # I6 (99 to 104)
            header['record_length_oat'] = int(first_line[105:111].strip())  # I6 (105 to 110)
            header['attitude_source'] = int(first_line[111:112].strip())  # I1 (111)
            header['mission_phase'] = int(first_line[112:113].strip())  # I1 (112)
            # Spare is from 113 onwards, not explicitly parsed.

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
    """Parses the OAT data file using fixed-width parsing."""
    data_records = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                line_stripped = line.strip()
                if not line_stripped.startswith('ORBTATTD'):
                    continue  # Skip header or empty lines if any

                try:
                    current_idx = 0
                    record = {}

                    record['record_type'] = line_stripped[current_idx: current_idx + 8].strip()
                    current_idx += 8

                    record['physical_record_num'] = int(line_stripped[current_idx: current_idx + 6].strip())
                    current_idx += 6

                    record['block_length_bytes'] = int(line_stripped[current_idx: current_idx + 4].strip())
                    current_idx += 4

                    # UTC time: 7I4 (28 bytes)
                    utc_raw_str = line_stripped[current_idx: current_idx + 28]
                    record['utc_time'] = parse_utc_time_from_parts(extract_time_parts_from_28char_string(utc_raw_str))
                    current_idx += 28

                    record['lunar_pos_xyz_j2000_earth_kms'] = np.array([
                        float(line_stripped[current_idx: current_idx + 20].strip()),  # F20.6
                        float(line_stripped[current_idx + 20: current_idx + 40].strip()),
                        float(line_stripped[current_idx + 40: current_idx + 60].strip())
                    ])
                    current_idx += 60  # 3 * 20

                    record['satellite_pos_xyz_j2000_kms'] = np.array([
                        float(line_stripped[current_idx: current_idx + 20].strip()),  # F20.6
                        float(line_stripped[current_idx + 20: current_idx + 40].strip()),
                        float(line_stripped[current_idx + 40: current_idx + 60].strip())
                    ])
                    current_idx += 60  # 3 * 20

                    record['satellite_vel_xyz_kms_sec'] = np.array([
                        float(line_stripped[current_idx: current_idx + 12].strip()),  # F12.6
                        float(line_stripped[current_idx + 12: current_idx + 24].strip()),
                        float(line_stripped[current_idx + 24: current_idx + 36].strip())
                    ])
                    current_idx += 36  # 3 * 12

                    record['sc_attitude_q_inertial_to_body'] = np.array([  # 4F14.10 (56 bytes)
                        float(line_stripped[current_idx: current_idx + 14].strip()),
                        float(line_stripped[current_idx + 14: current_idx + 28].strip()),
                        float(line_stripped[current_idx + 28: current_idx + 42].strip()),
                        float(line_stripped[current_idx + 42: current_idx + 56].strip())
                    ])
                    current_idx += 56

                    record['q_earth_fixed_iau'] = np.array([  # 4F14.10 (56 bytes)
                        float(line_stripped[current_idx: current_idx + 14].strip()),
                        float(line_stripped[current_idx + 14: current_idx + 28].strip()),
                        float(line_stripped[current_idx + 28: current_idx + 42].strip()),
                        float(line_stripped[current_idx + 42: current_idx + 56].strip())
                    ])
                    current_idx += 56

                    record['q_lunar_fixed_iau'] = np.array([  # 4F14.10 (56 bytes)
                        float(line_stripped[current_idx: current_idx + 14].strip()),
                        float(line_stripped[current_idx + 14: current_idx + 28].strip()),
                        float(line_stripped[current_idx + 28: current_idx + 42].strip()),
                        float(line_stripped[current_idx + 42: current_idx + 56].strip())
                    ])
                    current_idx += 56

                    record['sub_satellite_lat_deg'] = float(
                        line_stripped[current_idx: current_idx + 14].strip())  # F14.8
                    current_idx += 14
                    record['sub_satellite_lon_deg'] = float(
                        line_stripped[current_idx: current_idx + 14].strip())  # F14.8
                    current_idx += 14
                    record['solar_azimuth_deg'] = float(line_stripped[current_idx: current_idx + 14].strip())  # F14.8
                    current_idx += 14
                    record['solar_elevation_deg'] = float(line_stripped[current_idx: current_idx + 14].strip())  # F14.8
                    current_idx += 14
                    record['latitude_deg'] = float(line_stripped[current_idx: current_idx + 14].strip())  # F14.8
                    current_idx += 14
                    record['longitude_deg'] = float(line_stripped[current_idx: current_idx + 14].strip())  # F14.8
                    current_idx += 14
                    record['satellite_altitude_kms'] = float(
                        line_stripped[current_idx: current_idx + 12].strip())  # F12.3
                    current_idx += 12
                    record['roll_vel_angle_deg'] = float(line_stripped[current_idx: current_idx + 12].strip())  # F12.3
                    current_idx += 12
                    record['eclipse_status'] = int(line_stripped[current_idx: current_idx + 1].strip())  # I1
                    current_idx += 1
                    record['emission_angle_deg'] = float(line_stripped[current_idx: current_idx + 9].strip())  # F9.3
                    current_idx += 9
                    record['sun_angle_neg_yaw_phase_deg'] = float(
                        line_stripped[current_idx: current_idx + 9].strip())  # F9.3
                    current_idx += 9
                    record['yaw_nadir_angle_deg'] = float(line_stripped[current_idx: current_idx + 9].strip())  # F9.3
                    current_idx += 9
                    record['slant_range_km'] = float(line_stripped[current_idx: current_idx + 10].strip())  # F10.3
                    current_idx += 10
                    record['orbit_no'] = int(line_stripped[current_idx: current_idx + 5].strip())  # I5
                    current_idx += 5
                    record['solar_zenith_angle_deg'] = float(
                        line_stripped[current_idx: current_idx + 9].strip())  # F9.3
                    current_idx += 9
                    record['fov_vel_angle_deg'] = float(line_stripped[current_idx: current_idx + 9].strip())  # F9.3
                    current_idx += 9
                    record['x_yaw_angle_deg'] = float(line_stripped[current_idx: current_idx + 16].strip())  # F16.8
                    current_idx += 16
                    record['y_roll_angle_deg'] = float(line_stripped[current_idx: current_idx + 16].strip())  # F16.8
                    current_idx += 16
                    record['z_pitch_angle_deg'] = float(line_stripped[current_idx: current_idx + 16].strip())  # F16.8
                    current_idx += 16
                    # Spare is 41 bytes, not parsed. Check total length.
                    # assert current_idx + 41 == 628, f"OAT line {line_num} parsing length mismatch: {current_idx+41} vs 628"

                    data_records.append(record)
                except (ValueError, IndexError) as e:
                    print(f"Skipping malformed OAT line {line_num + 1}: {line_stripped} (Error: {e})")
        print(f"Parsed {len(data_records)} records from OAT file.")
        return data_records
    except FileNotFoundError:
        print(f"Error: OAT file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error parsing OAT file {filepath}: {e}")
        return []


def parse_spm_file(filepath):
    """Parses the SPM (Sun Parameter) file using fixed-width parsing."""
    data_records = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                line_stripped = line.strip()
                if not line_stripped.startswith('ORBTATTD'):
                    continue  # Skip non-data lines

                try:
                    current_idx = 0
                    record = {}

                    record['record_type'] = line_stripped[current_idx: current_idx + 8].strip()
                    current_idx += 8

                    record['physical_record_num'] = int(line_stripped[current_idx: current_idx + 6].strip())
                    current_idx += 6

                    record['block_length_bytes'] = int(line_stripped[current_idx: current_idx + 4].strip())
                    current_idx += 4

                    # UTC time: 7I4 (28 bytes)
                    utc_raw_str = line_stripped[current_idx: current_idx + 28]
                    record['utc_time'] = parse_utc_time_from_parts(extract_time_parts_from_28char_string(utc_raw_str))
                    current_idx += 28

                    record['satellite_pos_x_kms'] = float(line_stripped[current_idx: current_idx + 20].strip())  # F20.6
                    current_idx += 20
                    record['satellite_pos_y_kms'] = float(line_stripped[current_idx: current_idx + 20].strip())  # F20.6
                    current_idx += 20
                    record['satellite_pos_z_kms'] = float(line_stripped[current_idx: current_idx + 20].strip())  # F20.6
                    current_idx += 20

                    record['satellite_vel_x_kms_sec'] = float(
                        line_stripped[current_idx: current_idx + 12].strip())  # F12.6
                    current_idx += 12
                    record['satellite_vel_y_kms_sec'] = float(
                        line_stripped[current_idx: current_idx + 12].strip())  # F12.6
                    current_idx += 12
                    record['satellite_vel_z_kms_sec'] = float(
                        line_stripped[current_idx: current_idx + 12].strip())  # F12.6
                    current_idx += 12

                    record['phase_angle_deg'] = float(line_stripped[current_idx: current_idx + 9].strip())  # F9.3
                    current_idx += 9
                    record['sun_aspect_deg'] = float(line_stripped[current_idx: current_idx + 9].strip())  # F9.3
                    current_idx += 9
                    record['sun_azimuth_deg'] = float(line_stripped[current_idx: current_idx + 9].strip())  # F9.3
                    current_idx += 9
                    record['sun_elevation_deg'] = float(line_stripped[current_idx: current_idx + 9].strip())  # F9.3
                    current_idx += 9
                    record['orbit_limb_direction'] = int(line_stripped[current_idx: current_idx + 1].strip())  # I1
                    current_idx += 1
                    # Spare is 70 bytes, not parsed. Check total length.
                    # assert current_idx + 70 == 249, f"SPM line {line_num} parsing length mismatch: {current_idx+70} vs 249"

                    data_records.append(record)
                except (ValueError, IndexError) as e:
                    print(f"Skipping malformed SPM line {line_num + 1}: {line_stripped} (Error: {e})")
        print(f"Parsed {len(data_records)} records from SPM file.")
        return data_records
    except FileNotFoundError:
        print(f"Error: SPM file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error parsing SPM file {filepath}: {e}")
        return []


def parse_lbr_file(filepath):
    """Parses the LBR (Liberation Angle) file using fixed-width parsing."""
    data_records = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                line_stripped = line.strip()
                if not line_stripped.startswith('ORBTATTD'):
                    continue  # Skip non-data lines

                try:
                    current_idx = 0
                    record = {}

                    record['record_type'] = line_stripped[current_idx: current_idx + 8].strip()
                    current_idx += 8

                    record['physical_record_num'] = int(line_stripped[current_idx: current_idx + 6].strip())
                    current_idx += 6

                    record['block_length_bytes'] = int(line_stripped[current_idx: current_idx + 4].strip())
                    current_idx += 4

                    # UTC time: 7I4 (28 bytes)
                    utc_raw_str = line_stripped[current_idx: current_idx + 28]
                    record['utc_time'] = parse_utc_time_from_parts(extract_time_parts_from_28char_string(utc_raw_str))
                    current_idx += 28

                    record['satellite_pos_x_kms'] = float(line_stripped[current_idx: current_idx + 20].strip())  # F20.6
                    current_idx += 20
                    record['satellite_pos_y_kms'] = float(line_stripped[current_idx: current_idx + 20].strip())  # F20.6
                    current_idx += 20
                    record['satellite_pos_z_kms'] = float(line_stripped[current_idx: current_idx + 20].strip())  # F20.6
                    current_idx += 20

                    record['satellite_vel_x_kms_sec'] = float(
                        line_stripped[current_idx: current_idx + 12].strip())  # F12.6
                    current_idx += 12
                    record['satellite_vel_y_kms_sec'] = float(
                        line_stripped[current_idx: current_idx + 12].strip())  # F12.6
                    current_idx += 12
                    record['satellite_vel_z_kms_sec'] = float(
                        line_stripped[current_idx: current_idx + 12].strip())  # F12.6
                    current_idx += 12

                    record['liberation_angle_phi'] = float(
                        line_stripped[current_idx: current_idx + 16].strip())  # F16.8
                    current_idx += 16
                    record['liberation_angle_psi'] = float(
                        line_stripped[current_idx: current_idx + 16].strip())  # F16.8
                    current_idx += 16
                    record['liberation_angle_theta'] = float(
                        line_stripped[current_idx: current_idx + 16].strip())  # F16.8
                    current_idx += 16

                    record['liberation_angle_phi_rate'] = float(
                        line_stripped[current_idx: current_idx + 12].strip())  # F12.6
                    current_idx += 12
                    record['liberation_angle_psi_rate'] = float(
                        line_stripped[current_idx: current_idx + 12].strip())  # F12.6
                    current_idx += 12
                    record['liberation_angle_theta_rate'] = float(
                        line_stripped[current_idx: current_idx + 12].strip())  # F12.6
                    current_idx += 12
                    # Spare is 32 bytes, not parsed. Check total length.
                    # assert current_idx + 32 == 258, f"LBR line {line_num} parsing length mismatch: {current_idx+32} vs 258"

                    data_records.append(record)
                except (ValueError, IndexError) as e:
                    print(f"Skipping malformed LBR line {line_num + 1}: {line_stripped} (Error: {e})")
        print(f"Parsed {len(data_records)} records from LBR file.")
        return data_records
    except FileNotFoundError:
        print(f"Error: LBR file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error parsing LBR file {filepath}: {e}")
        return []


# --- 4. Camera Intrinsic Parameters (PLACEHOLDERS - YOU MUST REPLACE) ---
# These values are CRITICAL for accurate DEM generation.
# You need to find the calibration report or specifications for the Chandrayaan-2 camera
# (TMC, TMC-2, IIRS, or OHRC) that took the 'moonframe.png' image.

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

oath_header = parse_oath_header(OATH_PATH)
oat_data = parse_oat_file(OAT_PATH)
spm_data = parse_spm_file(SPM_PATH)
lbr_data = parse_lbr_file(LBR_PATH)

if not oath_header:
    print("Failed to parse OATH header. Exiting.")
    exit()

if not oat_data or not spm_data or not lbr_data:
    print("Error: Could not parse enough data from OAT, SPM or LBR files. Exiting.")
    exit()

# Determine the image acquisition time.
# IMPORTANT: In a real scenario, this would come from the image's metadata (EXIF, sidecar file).
# For now, we'll use the timestamp of the first OAT record as a proxy for the image's time.
image_acquisition_time = oat_data[0]['utc_time']
print(f"\nAssuming image acquisition time is: {image_acquisition_time} (from first OAT record)")


def find_closest_record(target_time, records):
    """Finds the record in a list of records with the closest UTC time to target_time."""
    if not records:
        return None

    closest_record = None
    min_time_diff = timedelta.max  # Initialize with a very large timedelta

    for record in records:
        if 'utc_time' in record and isinstance(record['utc_time'], datetime):
            time_diff = abs(target_time - record['utc_time'])
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_record = record
    return closest_record


relevant_oat_record = find_closest_record(image_acquisition_time, oat_data)
relevant_spm_record = find_closest_record(image_acquisition_time, spm_data)
relevant_lbr_record = find_closest_record(image_acquisition_time, lbr_data)

if not relevant_oat_record or not relevant_spm_record or not relevant_lbr_record:
    print("Error: Could not find relevant time-matched records. Exiting.")
    exit()

print(f"\nUsing OAT record from: {relevant_oat_record['utc_time']}")
print(f"Using SPM record from: {relevant_spm_record['utc_time']}")
print(f"Using LBR record from: {relevant_lbr_record['utc_time']}")

# --- 6. Calculate Illumination and Viewing Geometry ---

# Solar Incidence Angle = 90 - Sun Elevation Angle (from SPM file format description)
sun_elevation_deg = relevant_spm_record['sun_elevation_deg']
sun_azimuth_deg = relevant_spm_record['sun_azimuth_deg']

# Warn if sun elevation/azimuth are out of physical range
if not (-90 <= sun_elevation_deg <= 90):
    print(f"WARNING: Unphysical sun_elevation_deg detected from SPM: {sun_elevation_deg:.2f} degrees. Clipping.")
    sun_elevation_deg = np.clip(sun_elevation_deg, -90.0, 90.0)
if not (0 <= sun_azimuth_deg <= 360):
    print(f"WARNING: Unphysical sun_azimuth_deg detected from SPM: {sun_azimuth_deg:.2f} degrees. Modulo 360.")
    sun_azimuth_deg = sun_azimuth_deg % 360.0

solar_incidence_angle_deg = 90.0 - sun_elevation_deg
print(f"Solar Elevation (from SPM): {sun_elevation_deg:.6f} degrees (expected: 11.246643)")
print(f"Sun Azimuth (from SPM): {sun_azimuth_deg:.6f} degrees (expected: 302.523443)")
print(f"Solar Incidence Angle: {solar_incidence_angle_deg:.2f} degrees")

# Sun Azimuth and Elevation from SPM
sun_azimuth_rad = np.deg2rad(sun_azimuth_deg)
sun_elevation_rad = np.deg2rad(sun_elevation_deg)

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
# For a nadir-looking camera, the viewing vector points from the surface *to* the camera.
# In the local terrain frame (X-East, Y-North, Z-Up), the camera is "above" the surface.
# So, the viewing vector from the surface *to the camera* would be [0, 0, 1] (straight up).
view_vector_local_terrain = np.array([0, 0, 1])

print(f"Sun Vector (Local Terrain Frame): {sun_vector_local_terrain}")
print(f"View Vector (Local Terrain Frame): {view_vector_local_terrain}")


# --- 7. Photoclinometry (Shape-from-Shading) Core Logic ---

def lommel_seeliger_intensity(normal, L, V, albedo=0.1):
    """
    Calculates expected intensity for a given normal using a simplified Lommel-Seeliger model.
    normal, L, V are unit vectors. albedo is typically ~0.1 for lunar regolith.
    """

    # Ensure all vectors are unit vectors (handle near-zero for robustness)
    normal_unit = normal / (np.linalg.norm(normal) + 1e-9)
    L_unit = L / (np.linalg.norm(L) + 1e-9)
    V_unit = V / (np.linalg.norm(V) + 1e-9)

    # Cosine of incidence angle (N.L)
    cos_i = np.dot(normal_unit, L_unit)

    # Cosine of emission angle (N.V)
    # V points from surface to viewer.
    cos_e = np.dot(normal_unit, V_unit)

    # If cos_i < 0, surface is in shadow.
    # If cos_e < 0, surface is facing away from the camera.
    # Clip to 0 to represent no light/visibility
    cos_i = np.clip(cos_i, 0.0, 1.0)
    cos_e = np.clip(cos_e, 0.0, 1.0)

    if (cos_i + cos_e) == 0:
        return 0.0

    intensity = albedo * (cos_i / (cos_i + cos_e))
    return intensity


def photoclinometry_iterative_sfs(image_intensity, L, V, initial_dem, iterations, learning_rate, convergence_threshold):
    """
    Conceptual iterative Shape-from-Shading (SfS) algorithm.
    This is a simplification of a real SfS solver.
    """
    dem = initial_dem.copy()
    rows, cols = image_intensity.shape

    albedo = 0.1  # Simple albedo guess for Moon

    print(f"Starting {iterations} iterations of conceptual SfS...")

    last_max_abs_error = np.inf

    for i in range(iterations):
        # 1. Compute current surface normals (N) from the DEM
        # Gradients dz_dy, dz_dx (change in Z for y and x directions)
        dz_dy, dz_dx = np.gradient(dem)

        # Normals N = [-dz/dx, -dz/dy, 1] / sqrt( (dz/dx)^2 + (dz/dy)^2 + 1 )
        # Stack gradients to form normal vectors for each pixel
        normals_map_unnormalized = np.stack([-dz_dx, -dz_dy, np.ones_like(dem)], axis=-1)

        # Normalize each normal vector, add small epsilon to avoid division by zero
        norm = np.linalg.norm(normals_map_unnormalized, axis=-1, keepdims=True)
        normals_map = normals_map_unnormalized / (norm + 1e-9)  # Add epsilon here

        # 2. Compute expected image intensity (I_predicted) based on current normals and reflectance model
        I_predicted = np.zeros_like(image_intensity)
        # Vectorized dot products for efficiency
        # L and V are (3,) vectors, normals_map is (H, W, 3)
        # Reshape L and V for broadcasting: (1, 1, 3)
        L_reshaped = L[np.newaxis, np.newaxis, :]
        V_reshaped = V[np.newaxis, np.newaxis, :]

        cos_i_map = np.sum(normals_map * L_reshaped, axis=-1)
        cos_e_map = np.sum(normals_map * V_reshaped, axis=-1)

        # Clip values to valid range for photometric model
        cos_i_map_clipped = np.clip(cos_i_map, 0.0, 1.0)
        cos_e_map_clipped = np.clip(cos_e_map, 0.0, 1.0)

        denominator = cos_i_map_clipped + cos_e_map_clipped
        # Handle division by zero where denominator is very small
        I_predicted = np.where(denominator > 1e-9, albedo * (cos_i_map_clipped / denominator), 0.0)

        # Scale predicted intensity to match original range (for meaningful error calculation)
        min_I_pred, max_I_pred = I_predicted.min(), I_predicted.max()
        if max_I_pred - min_I_pred > 1e-9:  # Avoid division by zero for flat predictions
            I_predicted_normalized = (I_predicted - min_I_pred) / (max_I_pred - min_I_pred)
        else:
            I_predicted_normalized = np.zeros_like(I_predicted)  # All zeros if prediction is flat

        # 3. Compute difference (error)
        error = image_intensity - I_predicted_normalized

        # 4. Update DEM based on error (gradient descent type update)
        dem += learning_rate * error

        # Optional: Apply smoothing to the DEM to prevent noise amplification
        dem = gaussian_filter(dem, sigma=0.5)

        current_max_abs_error = np.max(np.abs(error))
        if (i + 1) % 10 == 0 or i == iterations - 1:
            print(f"  Iteration {i + 1}/{iterations}, Max Absolute Error: {current_max_abs_error:.4f}")

        # Check for convergence
        if abs(last_max_abs_error - current_max_abs_error) < convergence_threshold and i > 0:
            print(f"  Converged after {i + 1} iterations.")
            break
        last_max_abs_error = current_max_abs_error

    print("Conceptual SfS iterations complete.")
    return dem


# --- 8. DEM Generation ---

# Initial DEM guess: a flat surface at a reference altitude.
# We will use the satellite altitude from OAT for a rough reference.
initial_altitude_km = relevant_oat_record['satellite_altitude_kms']  # This is altitude *above the Moon's mean radius*
initial_dem_guess_m = np.full(image_array.shape, initial_altitude_km * 1000.0, dtype=np.float32)

# Run the iterative SfS (conceptual)
dem_relative_from_sfs = photoclinometry_iterative_sfs(
    image_normalized,
    sun_vector_local_terrain,
    view_vector_local_terrain,
    initial_dem_guess_m,  # Pass a base altitude guess
    iterations=NUM_SFS_ITERATIONS,
    learning_rate=SFS_LEARNING_RATE,
    convergence_threshold=CONVERGENCE_THRESHOLD
)

# --- 9. Final Scaling and Georeferencing (Conceptual) ---
# The DEM from SfS is *relative* height variations.
# To make it an absolute DEM in meters and connect it to lunar coordinates:
# 1. Scale the relative DEM:
#    The range of values in `dem_relative_from_sfs` needs to be mapped to a realistic
#    height range for lunar terrain.
#    This scaling factor is hard to determine without ground truth or stereo.
#    Let's use a heuristic for now, mapping the normalized relative DEM to a
#    plausible lunar terrain height range (e.g., 50 meters of variation).

# Normalize the SFS output to 0-1
min_dem_sfs, max_dem_sfs = dem_relative_from_sfs.min(), dem_relative_from_sfs.max()
if max_dem_sfs - min_dem_sfs > 1e-9:  # Avoid division by zero
    dem_scaled_0_1 = (dem_relative_from_sfs - min_dem_sfs) / (max_dem_sfs - min_dem_sfs)
else:
    dem_scaled_0_1 = np.zeros_like(dem_relative_from_sfs)  # All zeros if DEM is flat

# Map to a realistic height range in meters. (Guess, based on crater sizes)
# For example, assume the relief within the image is up to 100 meters.
assumed_relief_range_m = 100.0  # This is a major assumption.
absolute_dem_m = dem_scaled_0_1 * assumed_relief_range_m

# Add a base altitude to make it absolute.
# The 'satellite_altitude_kms' in OAT is already relative to the Moon's surface (radius).
# So, we can use the `initial_altitude_km` (which is the satellite's altitude above moon's mean radius)
# as a reference for the average height within the image, then modulate around that.
# The `initial_dem_guess_m` was already `initial_altitude_km * 1000.0`.
# The SfS process adjusts this initial flat surface.
# The 'absolute_dem_m' at this stage has a range of `assumed_relief_range_m`.
# To place it relative to the original `initial_dem_guess_m` we need to offset it.
# Let's consider the initial mean height as the reference for the middle of the scaled DEM.
mean_dem_sfs = (min_dem_sfs + max_dem_sfs) / 2
offset_from_mean_initial_m = (dem_relative_from_sfs - mean_dem_sfs) / (
            max_dem_sfs - min_dem_sfs + 1e-9) * assumed_relief_range_m / 2
absolute_dem_m = initial_dem_guess_m + offset_from_mean_initial_m

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
plt.imshow(absolute_dem_m, cmap='terrain')  # Using 'terrain' cmap for height visualization
plt.title(f'Conceptual Absolute DEM (meters)\nMin: {absolute_dem_m.min():.1f}m, Max: {absolute_dem_m.max():.1f}m')
plt.colorbar(label='Altitude (meters)')
plt.axis('off')

plt.tight_layout()
plt.show()

# --- 11. Save Results ---
# Normalize DEM for visualization (0-255 for PNG)
dem_min, dem_max = np.min(absolute_dem_m), np.max(absolute_dem_m)
if dem_max - dem_min > 0:
    dem_normalized_for_vis = (absolute_dem_m - dem_min) / (dem_max - dem_min) * 255.0
else:
    dem_normalized_for_vis = np.zeros_like(absolute_dem_m)  # Flat image if DEM is constant

output_image = Image.fromarray(dem_normalized_for_vis.astype(np.uint8))
output_image.save(OUTPUT_DEM_PNG)
print(f"Generated DEM image (normalized for visualization) saved to: {OUTPUT_DEM_PNG}")

# Save DEM as a NumPy array for quantitative analysis (in meters)
np.save(OUTPUT_DEM_NPY, absolute_dem_m)
print(f"Generated DEM NumPy array (in meters) saved to: {OUTPUT_DEM_NPY}")

print("\nSfS process completed. Check output files and plots.")
