import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Model & analysis settings --- #
interval = 1  # x_bin width (meters)

# Model geometry and linear fault dip range
model_begin = 20
model_width = 80
x_min = model_begin
x_max = model_begin + model_width  # 100
fault_dip_min = 20
fault_dip_max = 70

sediment_depth = 3
fault_wall_start = 40 * 0.4
threshold = 0.05
z_increase_threshold = 0.25

y_min = model_begin + 5
y_max = model_begin + 30

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

def process_file(filename, parsed_data, slip, strength, input_directory, output_directory, master_df):
    print(f"Processing file: {filename}, slip: {slip}")
    file_path = os.path.join(input_directory, filename)
    df = reformat_csv(file_path)
    df = df[(df['Y'] > y_min) & (df['Y'] < y_max)]
    df['x_bins'] = np.floor(df['X'] / interval) * interval
    results = []
    grouped = df.groupby('x_bins')
    for x_bin, group in grouped:
        # ------------ SKIP if already present in master_df -------------
        if (
            not master_df[
                (master_df['Slip'] == slip) &
                (master_df['x_bin'] == x_bin)
            ].empty
        ):
            continue
        # --------------------------------------------------------------

        bin_x = x_bin
        current_dip = round(fault_dip_for_x(bin_x), 2)
        sediment_height, est_v_disp, est_hw_disp, est_def_min, est_def_max = calculate_thresholds(
            slip, model_begin, sediment_depth, fault_wall_start, current_dip, threshold)
        group_sorted = group.sort_values(by='X')
        if group_sorted.empty:
            continue
        scarp_type = 'N/A'
        try:
            scarp_type = identify_scarp_type(
                group_sorted, slip, current_dip, model_begin, sediment_depth, threshold,
                est_v_disp, est_def_min, est_def_max, sediment_height
            )
        except TypeError:
            print(f" Type Error: {filename}")

        # PREPENDING FILENAME for all results
        base_row = [filename] + parsed_data

        if scarp_type == 'Pressure_Ridge':
            results.append(handle_pressure_ridge(group_sorted, x_bin, base_row, sediment_height, est_def_min, est_def_max, current_dip))
        elif scarp_type == 'Monoclinal':
            results.append(handle_monoclinal(group_sorted, x_bin, base_row, sediment_height, est_def_min, est_def_max, current_dip))
        elif scarp_type == 'Simple':
            results.append(handle_simple(group_sorted, x_bin, base_row, sediment_height, est_v_disp, est_def_min, est_def_max, current_dip))

        # optional plotting retained as in your original script...
        results_df = pd.DataFrame(results, columns=[
            "Filename", "Oblique", "Model", "Grain_Size", "coh", "dip", "FS", "Slip",
            "Scarp_Type", "x_bin",
            "scarp_height", "scarp_height_y", "adjusted_scarp_height",
            "dzw_min_y", "dzw_min_z", "dzw_max_y", "dzw_max_z", "dzw",
            "scarp_dip", "Us_Ud", "fault_dip"
        ])
        current_results = results_df[results_df['x_bin'] == x_bin]
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
            s = group_sorted['Radius']*75
        else:
            s = 8
        sc = ax.scatter(y, z, facecolors='grey', edgecolors='black', linewidths=0.25, alpha=0.25, s=s)
        sc3 = ax.scatter(current_results['dzw_min_y'], current_results['dzw_min_z']+0.5, color='red', label='DZW Min',alpha=1, s=10, marker='v',zorder=2)
        sc4 = ax.scatter(current_results['dzw_max_y'], current_results['dzw_max_z']+0.5, color='blue', label='DZW Max',alpha=1, s=10, marker='v',zorder=2)
        sc5 = ax.scatter(current_results['scarp_height_y'], current_results['scarp_height']+0.5, color='black', label='Scarp Height',alpha=1, s=10, marker='v', zorder=2)
        line2_x = [dzw_min_y_value, dzw_max_y_value]
        line2_y = [scarp_height_value + 1, scarp_height_value + 1]
        ax.plot(line2_x, line2_y, color='black', linestyle='solid', linewidth=1, label='DZW')
        label_text = f'DZW: {dzw_value:.2f} m' if dzw_value != 'NA' else 'DZW: NA'
        ax.text((line2_x[0] + line2_x[1]) / 2, (line2_y[0] + line2_y[1]) / 2 + 0.5, label_text, color='black', fontsize=9,ha='center', va='bottom', bbox=dict(facecolor='white', alpha=1, edgecolor='none'))
        if scarp_type == 'Simple':
            line1_x = [scarp_height_y_value, dzw_max_y_value]
            line1_y = [scarp_height_value - 0.5, dzw_max_z_value - 0.5]
            ax.plot(line1_x, line1_y, color='red', linestyle='dotted', linewidth=1, label='Scarp Dip')
        else:
            line1_x = [scarp_height_y_value, dzw_max_y_value]
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
    return results

def identify_scarp_type(group, slip, fault_dip, model_begin, sediment_depth, threshold, estimated_vertical_displacement, estimated_deformation_zone_min_value, estimated_deformation_zone_max_value, sediment_height):
    max_z = group['Z'].max()
    threshold_value = (model_begin + sediment_depth) + slip * np.sin(np.radians(fault_dip)) + 0.20
    group_sorted_desc = group.sort_values(by='Y', ascending=False)
    filtered_z_range = group_sorted_desc[(group_sorted_desc['Z'] > estimated_deformation_zone_min_value - threshold*4) &
                                         (group_sorted_desc['Z'] < estimated_deformation_zone_min_value + threshold*4)]
    max_y_value = filtered_z_range['Y'].max()
    first_decrease = filtered_z_range[filtered_z_range['Y'] == max_y_value]
    dzw_min_y = first_decrease.iloc[0]['Y'] if not first_decrease.empty else None
    dzw_min_z = first_decrease.iloc[0]['Z'] if not first_decrease.empty else None
    scarp_height_y = dzw_min_y
    scarp_height = dzw_min_z
    tolerance = 1
    group_sorted_asc = group.sort_values(by='Y', ascending=True)
    filtered_height = group_sorted_asc[
        (group_sorted_asc['Z'] > model_begin + slip * np.sin(np.radians(fault_dip)) - tolerance) &
        (group_sorted_asc['Z'] < model_begin + slip * np.sin(np.radians(fault_dip)) + tolerance)]
    dzw_max_y = filtered_height['Y'].max() if not filtered_height.empty else None
    if dzw_max_y is None:
        filtered_height = group_sorted_asc[
        (group_sorted_asc['Z'] > model_begin + sediment_depth - threshold * 4) &
        (group_sorted_asc['Z'] < model_begin + sediment_depth + threshold * 4)]
        dzw_max_y = filtered_height['Y'].max()
    dzw_max_y_threshold = scarp_height_y + (4 * threshold) if scarp_height_y is not None else None
    if max_z > threshold_value:
        return 'Pressure_Ridge'
    elif dzw_max_y is not None and dzw_max_y_threshold is not None and dzw_max_y > dzw_max_y_threshold:
        return 'Monoclinal'
    elif dzw_max_y is not None and scarp_height_y is not None and dzw_max_y < scarp_height_y:
        return 'Simple'
    else:
        return 'N/A'

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
    scarp_dip = np.degrees(np.arctan(abs(dzw_max_z - scarp_height) / abs(dzw_max_y - scarp_height_y))) if dzw_max_y is not None and scarp_height_y is not None and abs(dzw_max_y - scarp_height_y) > 0 else None
    Us_Ud = scarp_height - dzw_max_z if dzw_max_z is not None else None
    return adjusted_scarp_height, dzw, scarp_dip, Us_Ud

def handle_pressure_ridge(group, x_bin, parsed_data, sediment_height, estimated_deformation_zone_min_value, estimated_deformation_zone_max_value, current_dip):
    scarp_height = group['Z'].max()
    scarp_height_y = group[group['Z'] == scarp_height].iloc[0]['Y']
    group_sorted_asc = group.sort_values(by='Y', ascending=True)
    first_increase = group_sorted_asc[(group_sorted_asc['Z'] > estimated_deformation_zone_min_value + threshold * 4) &
                                      (group_sorted_asc['Y'] < scarp_height_y)]
    dzw_min_y = first_increase.iloc[0]['Y'] if not first_increase.empty else None
    dzw_min_z = first_increase.iloc[0]['Z'] if not first_increase.empty else None
    dzw_max_y, dzw_max_z = None, None
    if scarp_height_y is not None:
        group_sorted_desc = group.sort_values(by='Y', ascending=False)
        last_decrease = group_sorted_desc[(group_sorted_desc['Z'] > sediment_height + threshold) & (group_sorted_desc['Y'] > scarp_height_y)]
        dzw_max_y = last_decrease.iloc[0]['Y'] if not last_decrease.empty else None
        dzw_max_z = last_decrease.iloc[0]['Z'] if not last_decrease.empty else None
    adjusted_scarp_height, dzw, scarp_dip, Us_Ud = calculate_common_metrics(scarp_height, scarp_height_y, dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z)
    return parsed_data + ['Pressure_Ridge', x_bin,
         scarp_height, scarp_height_y, adjusted_scarp_height,
         dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z, dzw,
         scarp_dip, Us_Ud, current_dip]

def handle_monoclinal(group, x_bin, parsed_data, sediment_height, estimated_deformation_zone_min_value, estimated_deformation_zone_max_value, current_dip):
    group_sorted_desc = group.sort_values(by='Y', ascending=False)
    filtered_z_range = group_sorted_desc[(group_sorted_desc['Z'] > estimated_deformation_zone_min_value - threshold * 2) &
                                         (group_sorted_desc['Z'] < estimated_deformation_zone_min_value + threshold * 2)]
    if filtered_z_range.empty:
        return parsed_data + ['NA', x_bin, None, None, None, None, None, None, None, None, None, None, current_dip]
    max_y_value = filtered_z_range['Y'].max()
    first_decrease = filtered_z_range[filtered_z_range['Y'] == max_y_value]
    dzw_min_y = first_decrease.iloc[0]['Y'] if not first_decrease.empty else None
    dzw_min_z = first_decrease.iloc[0]['Z'] if not first_decrease.empty else None
    scarp_height_y = dzw_min_y
    scarp_height = dzw_min_z
    dzw_max_y, dzw_max_z = None, None
    if scarp_height_y is not None:
        last_decrease = group_sorted_desc[(group_sorted_desc['Z'] > sediment_height + threshold) &
                                          (group_sorted_desc['Y'] > scarp_height_y)]
        dzw_max_y = last_decrease.iloc[0]['Y'] if not last_decrease.empty else None
        dzw_max_z = last_decrease.iloc[0]['Z'] if not last_decrease.empty else None
    adjusted_scarp_height, dzw, scarp_dip, Us_Ud = calculate_common_metrics(scarp_height, scarp_height_y, dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z)
    return parsed_data + ['Monoclinal', x_bin,
         scarp_height, scarp_height_y, adjusted_scarp_height,
         dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z, dzw,
         scarp_dip, Us_Ud, current_dip]

def handle_simple(group, x_bin, parsed_data, sediment_height, estimated_vertical_displacement, estimated_deformation_zone_min_value, estimated_deformation_zone_max_value, current_dip):
    group_sorted_desc = group.sort_values(by='Y', ascending=False)
    filtered_z_range = group_sorted_desc[(group_sorted_desc['Z'] > estimated_deformation_zone_min_value - threshold) &
                                         (group_sorted_desc['Z'] < estimated_deformation_zone_min_value + threshold)]
    if filtered_z_range.empty:
        return parsed_data + ['NA', x_bin, None, None, None, None, None, None, None, None, None, None, current_dip]
    max_y_value = filtered_z_range['Y'].max()
    first_decrease = filtered_z_range[filtered_z_range['Y'] == max_y_value]
    dzw_min_y = first_decrease.iloc[0]['Y'] if not first_decrease.empty else None
    dzw_min_z = first_decrease.iloc[0]['Z'] if not first_decrease.empty else None
    scarp_height_y = dzw_min_y
    scarp_height = dzw_min_z
    top_fw = sediment_height
    group_sorted_asc = group.sort_values(by='Y', ascending=True)
    filtered_z_range2 = group_sorted_asc[(group_sorted_asc['Z'] > top_fw + 4*threshold) &
                                         (group_sorted_asc['Z'] < top_fw + 5*threshold)]
    max_y_value2 = filtered_z_range2['Y'].max()
    second_decrease = filtered_z_range2[filtered_z_range2['Y'] == max_y_value2]
    dzw_max_y = second_decrease.iloc[0]['Y'] if not second_decrease.empty else None
    dzw_max_z = second_decrease.iloc[0]['Z'] if not second_decrease.empty else None
    adjusted_scarp_height, dzw, scarp_dip, Us_Ud = calculate_common_metrics(scarp_height, scarp_height_y, dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z)
    return parsed_data + ['Simple', x_bin,
         scarp_height, scarp_height_y, adjusted_scarp_height,
         dzw_min_y, dzw_min_z, dzw_max_y, dzw_max_z, dzw,
         scarp_dip, Us_Ud, current_dip]

def main():
    process_single_file = False
    strengths = ['2.0e+06'] #,'CTU','Random']
    strength_types = ['ModerateWeak'] # ,'CTU','Random']

    for strength_type in strength_types:
        for strength in strengths:
            input_directory = f"/Users/kchiama/Harvard University Dropbox/SGER Group/Chiama_K/3Ddem/Analysis/csv/Case2/{strength_type}"
            output_csv = f"/Users/kchiama/Harvard University Dropbox/SGER Group/Chiama_K/3Ddem/Analysis/csv/SDC/Case2_D3_75x_{strength_type}_SDC.csv"
            output_directory = f"/Users/kchiama/Harvard University Dropbox/SGER Group/Chiama_K/3Ddem/Analysis/csv/figures/Case2_D3_75x_{strength_type}"
            os.makedirs(output_directory, exist_ok=True)

            if not os.path.exists(input_directory):
                print(f"Directory {input_directory} does not exist. Skipping...")
                continue

            filenames = [f for f in os.listdir(input_directory) if f.endswith('.csv') and os.path.isfile(os.path.join(input_directory, f))]

            if process_single_file:
                file_name = f"D3_75x_{strength}_dip20_FS0.5_5.0slip.csv"
                if file_name in filenames:
                    filenames = [file_name]
                else:
                    print(f"File {file_name} not found in the directory.")
                    continue

            filenames_with_slip = []
            for filename in filenames:
                parsed_data = parse_filename(filename)
                slip = float(parsed_data[6]) if parsed_data[6] else 0
                filenames_with_slip.append((filename, parsed_data, slip))

            filenames_with_slip.sort(key=lambda x: x[2], reverse=True)  # Descending by slip, highest first

            headers = ["Filename", "Oblique", "Model", "Grain_Size", "coh", "dip", "FS", "Slip",
                        "Scarp_Type", "x_bin",
                        "scarp_height", "scarp_height_y", "adjusted_scarp_height",
                        "dzw_min_y", "dzw_min_z", "dzw_max_y", "dzw_max_z", "dzw",
                        "scarp_dip", "Us_Ud", "fault_dip"]

            if os.path.exists(output_csv):
                master_df = pd.read_csv(output_csv)
            else:
                master_df = pd.DataFrame(columns=headers)

            for filename, parsed_data, slip in filenames_with_slip:
                file_results = process_file(filename, parsed_data, slip, strength, input_directory, output_directory, master_df)
                if not file_results:
                    continue
                results_df = pd.DataFrame(file_results, columns=headers)

                # Append or update per (Slip, x_bin)
                for _, row in results_df.iterrows():
                    mask = (
                        (master_df['Slip'] == row['Slip']) &
                        (master_df['x_bin'] == row['x_bin'])
                    )
                    if master_df[mask].empty:
                        master_df = pd.concat([master_df, pd.DataFrame([row], columns=headers)], ignore_index=True)
                    else:
                        master_df.loc[mask, :] = row.values

                master_df.to_csv(output_csv, index=False)
                print(f"Processed slip={slip:.2f}")

            print(f"Data has been saved to {output_csv}")

if __name__ == '__main__':
    main()