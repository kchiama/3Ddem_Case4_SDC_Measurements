import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

interval = 1  # x_bin width (meters)
model_begin = 20
model_width = 50
x_min = model_begin
x_max = model_begin + model_width  # 100
fault_dip_min = 20
fault_dip_max = 70
sediment_depth = 3
fault_wall_start = 25
threshold = 0.05
z_increase_threshold = 0.25
colluvial_wedge_threshold = 0.20
y_min = model_begin + 5
y_max = model_begin + 60

continuous_fault_dip = False
# OBLIQUE CROP PARAMS
OBLIQUE_ANGLE_DEG = 30
OBLIQUE_WINDOW = 10.0
center_x = model_begin + 40 / 2
center_y = model_begin + 60 / 2

def crop_group_along_oblique(group, x_bin, center_x, center_y, oblique_deg, window=10.0):
    theta = np.radians(oblique_deg)
    y_ridge = center_y - (x_bin - center_x) * np.tan(theta)
    return group[np.abs(group['Y'] - y_ridge) < window].copy()

def fault_dip_for_x(x):
    x_clamped = max(min(x, x_max), x_min)
    progression = (x_clamped - x_min) / (x_max - x_min)
    return fault_dip_min + (fault_dip_max - fault_dip_min) * progression

def set_axes_equal_with_limits(ax, x_limits, y_limits):
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    plot_radius = 0.5 * max(x_range, y_range)
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    ax.set_xlim([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim([y_middle - plot_radius/2, y_middle + plot_radius/2])

def safe_value_extraction(series, default):
    return series.values[0] if not series.isnull().any() and len(series) > 0 else default

def calculate_thresholds(slip, model_begin, sediment_depth, fault_wall_start, fault_dip, threshold):
    sediment_height = model_begin + sediment_depth
    estimated_vertical_displacement = model_begin + slip * np.sin(np.radians(fault_dip))
    estimated_hanging_wall_displacement = model_begin + sediment_depth + slip * np.sin(np.radians(fault_dip))
    estimated_deformation_zone_min_value = model_begin + sediment_depth + slip * np.sin(np.radians(fault_dip))
    estimated_deformation_zone_max_value = model_begin + fault_wall_start + slip * np.cos(np.radians(fault_dip))
    return (sediment_height, estimated_vertical_displacement, estimated_hanging_wall_displacement, estimated_deformation_zone_min_value, estimated_deformation_zone_max_value)

def parse_filename(filename):
    name_without_extension = os.path.splitext(filename)[0]
    parts = name_without_extension.split('_')
    oblique = parts[0].replace('Oblique', '') if 'Oblique' in parts[0] else None
    model = parts[1] if parts[1].startswith('D') else None
    grain_size = parts[2] if len(parts) > 2 else None
    coh = parts[3].replace('coh', '') if len(parts) > 3 and parts[3].startswith('coh') else None
    dip = parts[4].replace('dip', '') if len(parts) > 4 and parts[4].startswith('dip') else None
    fs = parts[5].replace('FS', '') if len(parts) > 5 and parts[5].startswith('FS') else None
    slip = parts[6].replace('slip', '') if len(parts) > 6 and parts[6].endswith('slip') else None
    return [oblique, model, grain_size, coh, dip, fs, slip]

def reformat_csv(file_path):
    df = pd.read_csv(file_path)
    column_to_split = 'Position'
    def parse_column(val):
        try:
            values = val.strip('[]').split()
            return [float(x) for x in values]
        except (ValueError, SyntaxError):
            return [None, None, None]
    parsed_values = df[column_to_split].apply(parse_column)
    df[['X', 'Y', 'Z']] = pd.DataFrame(parsed_values.tolist(), index=df.index)
    df.drop(columns=[column_to_split], inplace=True)
    return df

def identify_best_scarp(
        group, slip, fault_dip, model_begin, sediment_depth, threshold,
        estimated_vertical_displacement, estimated_deformation_zone_min_value,
        estimated_deformation_zone_max_value, sediment_height, x_bin=None):

    max_z = group['Z'].max()
    threshold_value = (model_begin + sediment_depth) + slip * np.sin(np.radians(fault_dip)) + 0.10
    if max_z > threshold_value:
        scarp_height_y = group[group['Z'] == max_z]['Y'].values[0]
        scarp_height = max_z
        group_sorted_asc = group.sort_values(by='Y', ascending=True)
        threshold_for_ridge = estimated_deformation_zone_min_value + 2 * threshold
        first_increase = group_sorted_asc[
            (group_sorted_asc['Z'] > threshold_for_ridge) &
            (group_sorted_asc['Y'] < scarp_height_y)]
        dzw_min_y = first_increase.iloc[0]['Y'] if not first_increase.empty else None
        dzw_min_z = first_increase.iloc[0]['Z'] if not first_increase.empty else None
        group_sorted_desc = group.sort_values(by='Y', ascending=False)
        last_decrease = group_sorted_desc[
            (group_sorted_desc['Z'] > sediment_height + threshold) &
            (group_sorted_desc['Y'] > scarp_height_y)]
        dzw_max_y = last_decrease.iloc[0]['Y'] if not last_decrease.empty else None
        dzw_max_z = last_decrease.iloc[0]['Z'] if not last_decrease.empty else None
        return 'Pressure_Ridge', dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z, scarp_height_y, scarp_height

    group_sorted_desc = group.sort_values(by='Y', ascending=False)
    filtered_z_range = group_sorted_desc[
        (group_sorted_desc['Z'] > estimated_deformation_zone_min_value - threshold * 6) &
        (group_sorted_desc['Z'] < estimated_deformation_zone_min_value + threshold * 6)]
    dzw_min_y = filtered_z_range['Y'].max() if not filtered_z_range.empty else None
    dzw_min_z = filtered_z_range[filtered_z_range['Y'] == dzw_min_y]['Z'].values[0] if dzw_min_y is not None else None

    scarp_height_y = dzw_min_y
    scarp_height = dzw_min_z

    fw_level = model_begin + sediment_depth + colluvial_wedge_threshold
    mono_candidates = group[group['Z'] > fw_level]
    is_monoclinal = False
    dzw_max_y = None
    dzw_max_z = None

    if not mono_candidates.empty:
        dzw_max_y = mono_candidates['Y'].max()
        dzw_max_z = mono_candidates[mono_candidates['Y'] == dzw_max_y]['Z'].values[0]
        if dzw_min_y is not None and abs(dzw_max_y - dzw_min_y) > 0.3 and len(mono_candidates) >= 2:
            is_monoclinal = True

    if is_monoclinal:
        return 'Monoclinal', dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z, scarp_height_y, scarp_height
    ## --- SIMPLE scarp: DZW max is max Y fully above footwall block plus a buffer range ---
    footwall_z = model_begin + sediment_depth
    buffer_min = 0.25
    buffer_max = 0.5

    z_min = footwall_z + buffer_min

    # Compute both possible z_max options
    z_max_option1 = model_begin + buffer_max + slip * np.sin(np.radians(fault_dip))
    z_max_option2 = footwall_z + buffer_min + buffer_max
    z_max = max(z_max_option1, z_max_option2)

    simple_candidates = group[(group['Z'] > z_min) & (group['Z'] < z_max)]
    if not simple_candidates.empty:
        dzw_max_y = simple_candidates['Y'].max()
        dzw_max_z = simple_candidates[simple_candidates['Y'] == dzw_max_y]['Z'].values[0]
    else:
        dzw_max_y = np.nan
        dzw_max_z = np.nan

    return 'Simple', dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z, scarp_height_y, scarp_height

def calculate_common_metrics(scarp_height, scarp_height_y, dzw_min_y, dzw_min_x, dzw_max_y, dzw_max_z):
    adjusted_scarp_height = scarp_height - (model_begin + sediment_depth)
    dzw1 = abs(dzw_max_y - scarp_height_y) if dzw_max_y is not None and scarp_height_y is not None else None
    dzw2 = abs(dzw_max_y - dzw_min_y) if dzw_max_y is not None and dzw_min_y is not None else None
    if dzw1 is not None and dzw2 is not None:
        dzw = max(dzw1, dzw2)
    elif dzw1 is not None:
        dzw = dzw1
    else:
        dzw = dzw2
    scarp_dip = np.degrees(np.arctan(abs(dzw_max_z - scarp_height) / abs(dzw_max_y - scarp_height_y))) \
        if dzw_max_y is not None and scarp_height_y is not None and abs(dzw_max_y - scarp_height_y) > 0 else None
    Us_Ud = scarp_height - dzw_max_z if dzw_max_z is not None else None
    return adjusted_scarp_height, dzw, scarp_dip, Us_Ud

def process_file(filename, parsed_data, slip, strength_value, strength_label, input_directory, output_directory, output_csv, headers, fault_dip, continuous_fault_dip, completed_xbins_set):
    print(f"Processing file: {filename}, slip: {slip}")
    file_path = os.path.join(input_directory, filename)
    df = reformat_csv(file_path)
    df = df[(df['Y'] > y_min) & (df['Y'] < y_max)]
    df['x_bins'] = np.floor(df['X'] / interval) * interval

    grouped = df.groupby('x_bins')

    for x_bin, group in grouped:
        # ---- NEW: Skip bins already completed ----
        key = (filename, float(x_bin))
        if key in completed_xbins_set:
            # print(f"Skipping (filename={filename}, x_bin={x_bin}): already processed in output.")
            continue

        group = crop_group_along_oblique(group, x_bin, center_x, center_y, OBLIQUE_ANGLE_DEG, OBLIQUE_WINDOW)
        if group.empty:
            continue

        if continuous_fault_dip:
            current_dip = round(fault_dip_for_x(x_bin), 2)
        else:
            current_dip = float(fault_dip)

        sediment_height, est_v_disp, est_hw_disp, est_def_min, est_def_max = calculate_thresholds(
            slip, model_begin, sediment_depth, fault_wall_start, current_dip, threshold)
        group_sorted = group.sort_values(by='X')
        if group_sorted.empty:
            continue

        scarp_type, dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z, scarp_height_y, scarp_height = identify_best_scarp(
            group_sorted, slip, current_dip, model_begin, sediment_depth, threshold,
            est_v_disp, est_def_min, est_def_max, sediment_height,x_bin)

        if scarp_type == 'Pressure_Ridge':
            if dzw_max_y is not None and scarp_height_y is not None and abs(dzw_max_y - scarp_height_y) > 0:
                scarp_dip = np.degrees(np.arctan(abs(dzw_max_z - scarp_height) / abs(dzw_max_y - scarp_height_y)))
            else:
                scarp_dip = None
        else:
            adjusted_scarp_height, dzw, scarp_dip, Us_Ud = calculate_common_metrics(
                scarp_height, scarp_height_y, dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z)

        adjusted_scarp_height, dzw, *_ = calculate_common_metrics(
            scarp_height, scarp_height_y, dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z)
        Us_Ud = scarp_height - dzw_max_z if dzw_max_z is not None else None

        # Output must contain filename and x_bin
        result_fields = [filename, scarp_type, x_bin, scarp_height, scarp_height_y, adjusted_scarp_height,
                        dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z, dzw,
                        scarp_dip, Us_Ud, current_dip]
        result_fields_clean = [v if v is not None else np.nan for v in result_fields]

        row_df = pd.DataFrame([parsed_data + [strength_label, strength_value] + result_fields_clean], columns=headers)
        row_df.to_csv(output_csv, index=False, mode='a', header=False)

        current_results = row_df
        scarp_height_value = safe_value_extraction(current_results['scarp_height'], 0)
        scarp_height_y_value = safe_value_extraction(current_results['scarp_height_y'], 0)
        dzw_min_y_value = safe_value_extraction(current_results['dzw_min_y'], 0)
        dzw_min_z_value = safe_value_extraction(current_results['dzw_min_z'], 0)
        dzw_max_y_value = safe_value_extraction(current_results['dzw_max_y'], 0)
        dzw_max_z_value = safe_value_extraction(current_results['dzw_max_z'], 0)
        dzw_value = safe_value_extraction(current_results['dzw'], 'NA')
        scarp_dip_value = safe_value_extraction(current_results['scarp_dip'], 'NA')

        fig, ax = plt.subplots(figsize=(10, 5))
        y = group_sorted['Y']
        z = group_sorted['Z']
        if 'Radius' in group_sorted:
            s = group_sorted['Radius'] * 75
        else:
            s = 8
        ax.scatter(y, z, facecolors='grey', edgecolors='black', linewidths=0.25, alpha=0.25, s=s)
        ax.scatter(dzw_min_y_value, dzw_min_z_value+0.5, color='red', label='DZW Min',alpha=1, s=10, marker='v',zorder=2)
        ax.scatter(dzw_max_y_value, dzw_max_z_value+0.5, color='blue', label='DZW Max',alpha=1, s=10, marker='v',zorder=2)
        ax.scatter(scarp_height_y_value, scarp_height_value+0.5, color='black', label='Scarp Height',alpha=1, s=10, marker='v', zorder=2)
        line2_x = [dzw_min_y_value, dzw_max_y_value]
        line2_y = [scarp_height_value + 1, scarp_height_value + 1]
        ax.plot(line2_x, line2_y, color='black', linestyle='solid', linewidth=1, label='DZW')
        label_text = f'DZW: {dzw_value:.2f} m' if dzw_value != 'NA' else 'DZW: NA'
        ax.text((line2_x[0] + line2_x[1]) / 2, (line2_y[0] + line2_y[1]) / 2 + 0.5, label_text, color='black', fontsize=9,ha='center', va='bottom', bbox=dict(facecolor='white', alpha=1, edgecolor='none'))
        line1_x = [scarp_height_y_value, dzw_max_y_value]
        if scarp_type == 'Simple':
            line1_y = [scarp_height_value - 0.5, dzw_max_z_value - 0.5]
        else:
            line1_y = [scarp_height_value + 1, dzw_max_z_value + 1]
        ax.plot(line1_x, line1_y, color='red', linestyle='dotted', linewidth=1, label='Scarp Dip')
        label_text = f'Scarp Dip: {scarp_dip_value:.2f}º' if scarp_dip_value != 'NA' else 'Scarp Dip: NA'
        ax.text((line1_x[0] + line1_x[1]) / 2 + 1, (line1_y[0] + line1_y[1]) / 2, label_text, color='grey', fontsize=9,
            ha='left', va='bottom', bbox=dict(facecolor='white', alpha=1, edgecolor='none'))
        desired_x_limits = (y_min, y_max)
        desired_y_limits = (19, 30)
        set_axes_equal_with_limits(ax, desired_x_limits, desired_y_limits)
        ax.set_title(f"x_bin: {x_bin:.2f}, Dip: {current_dip}°, Slip: {slip}, Scarp Type: {scarp_type}")
        ax.set_xlabel("Y Position (m)")
        ax.set_ylabel("Z Position (m)")
        ax.legend()
        output_png_path = os.path.join(output_directory, f"{filename}_xbin{x_bin:.2f}.png")
        plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def main():
    cases = ['Case4']
    strength_types = ['Strong']
    fault_dips = [20]
    strength_map = {'Weak': '1.0e+06', 'Moderate': '3.0e+06', 'Strong': '5.0e+06', 'Random': '3.0e+06', 'CTU': '1.0e+06'}

    headers =  ["Oblique", "Model", "Grain_Size", "coh", "dip", "FS", "Slip",
                "Strength_Label", "Strength_Value", "filename",
                "Scarp_Type", "x_bin",
                "scarp_height", "scarp_height_y", "adjusted_scarp_height",
                "dzw_min_y", "dzw_min_z", "dzw_max_y", "dzw_max_z", "dzw",
                "scarp_dip", "Us_Ud", "fault_dip"]

    for case in cases:
        for strength_label in strength_types:
            strength_value = strength_map[strength_label]
            for fault_dip in fault_dips:
                print('Starting New Experiment...')
                print(50*'=')
                input_directory = f"/Users/kchiama/Harvard University Dropbox/SGER Group/Chiama_K/3Ddem/Analysis/csv/{case}/Oblique30/{strength_label}"
                output_csv = f"/Users/kchiama/Harvard University Dropbox/SGER Group/Chiama_K/3Ddem/Analysis/csv/SDC/{case}_Oblique30_D3_75x_{strength_label}_dip{fault_dip}_SDC.csv"
                output_directory = f"/Users/kchiama/Harvard University Dropbox/SGER Group/Chiama_K/3Ddem/Analysis/csv/figures/{case}_Oblique30_D3_75x_{strength_label}_dip{fault_dip}"
                os.makedirs(output_directory, exist_ok=True)
                if not os.path.exists(input_directory):
                    print(f"Directory {input_directory} does not exist. Skipping...")
                    continue

                # ---- KEY: Resume support ----
                completed_xbins_set = set()
                if os.path.exists(output_csv):
                    prev = pd.read_csv(output_csv)
                    if 'x_bin' in prev.columns and 'filename' in prev.columns:
                        completed_xbins_set = set((str(f), float(xb)) for f, xb in zip(prev['filename'], prev['x_bin']))
                    else:
                        completed_xbins_set = set()
                else:
                    pd.DataFrame(columns=headers).to_csv(output_csv, index=False, mode='w', header=True)

                filenames = [
                    f for f in os.listdir(input_directory)
                    if f.endswith('.csv')
                    and os.path.isfile(os.path.join(input_directory, f))
                    and f'dip{fault_dip}' in f
                    and f'coh{strength_value}' in f
                ]
                filenames_with_slip = []
                for filename in filenames:
                    parsed_data = parse_filename(filename)
                    slip = float(parsed_data[6]) if parsed_data[6] else 0
                    filenames_with_slip.append((filename, parsed_data, slip))
                filenames_with_slip.sort(key=lambda x: -x[2])
                for filename, parsed_data, slip in filenames_with_slip:
                    process_file(filename, parsed_data, slip, strength_value, strength_label,
                                input_directory, output_directory, output_csv, headers,
                                fault_dip, continuous_fault_dip, completed_xbins_set)
                print(f"Data has been saved to {output_csv}")
                
                plt.close('all')
                for var in ['df', 'group', 'grouped', 'group_sorted']:
                    try:
                        del globals()[var]
                    except Exception:
                        pass
                import gc; gc.collect()
                print('Cleared python space')

if __name__ == '__main__':
    main()
