"""
ENSO Table Visualization utilities.
Converts ENSO trajectory into styled year-month table with colors.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb


def create_enso_table(enso_traj, threshold=0.5, start_month=0):
    """
    Create a formatted ENSO table (years x 3-month seasons), aligned to the
    calendar so column j is always the season centered on calendar month j
    (DJF=Jan, JFM=Feb, ..., NDJ=Dec).

    Args:
        enso_traj (array): ENSO time series (already smooth / ONI)
        threshold (float): Threshold for coloring
        start_month (int): 0-based calendar month of the first value (0=Jan).
            With the env's month_offset, a rollout can start in any month, so we
            pad the front with `start_month` blanks (and the tail to fill the last
            row) so values land in their true calendar column. Blanks are NaN.

    Returns:
        pd.DataFrame: dataframe with ENSO values, NaN for out-of-range padding.
    """
    enso_data = np.asarray(enso_traj, dtype=float)
    n_months = 12

    start_month = int(start_month) % n_months
    # Pad front so data[0] lands in column `start_month`, and tail to a full row.
    pad_front = start_month
    total = pad_front + len(enso_data)
    pad_back = (-total) % n_months  # round up to a multiple of 12
    padded = np.concatenate([
        np.full(pad_front, np.nan),
        enso_data,
        np.full(pad_back, np.nan),
    ])
    reshaped_data = padded.reshape(-1, n_months)

    # Create DataFrame with 3-month season labels
    months = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
    df_enso = pd.DataFrame(reshaped_data, columns=months)
    df_enso.index.name = 'Year'
    df_enso.index = df_enso.index + 1

    return df_enso


def _find_peak_positions(df_enso, threshold=0.5):
    """
    Find (row, col) positions of peak values within each continuous ENSO run.

    Returns:
        set: Set of (row_idx, col_idx) tuples marking peak positions
    """
    flat = df_enso.values.flatten()
    n = len(flat)
    peaks = set()

    # Find El Niño peaks (max within each run above threshold)
    i = 0
    while i < n:
        if flat[i] > threshold:
            start = i
            while i < n and flat[i] > threshold:
                i += 1
            peak_idx = start + np.argmax(flat[start:i])
            peaks.add(divmod(peak_idx, df_enso.shape[1]))
        else:
            i += 1

    # Find La Niña peaks (min within each run below -threshold)
    i = 0
    while i < n:
        if flat[i] < -threshold:
            start = i
            while i < n and flat[i] < -threshold:
                i += 1
            peak_idx = start + np.argmin(flat[start:i])
            peaks.add(divmod(peak_idx, df_enso.shape[1]))
        else:
            i += 1

    return peaks


def style_enso_table(df_enso, threshold=0.5):
    """
    Apply styling to ENSO table. Peak values in each El Niño/La Niña run are bolded.

    Args:
        df_enso (pd.DataFrame): ENSO dataframe
        threshold (float): Threshold for coloring

    Returns:
        pd.io.formats.style.Styler: Styled dataframe
    """
    peaks = _find_peak_positions(df_enso, threshold)

    def style_func(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for i in range(len(df)):
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                if pd.isna(val):
                    styles.iloc[i, j] = 'background-color: white'  # calendar padding: blank
                    continue
                bold = 'font-weight: bold' if (i, j) in peaks else 'font-weight: normal'
                if val > threshold:
                    styles.iloc[i, j] = f'background-color: #FFB3B3; color: red; {bold}'
                elif val < -threshold:
                    styles.iloc[i, j] = f'background-color: #B3D9FF; color: blue; {bold}'
                else:
                    styles.iloc[i, j] = f'background-color: #E8E8E8; color: black; {bold}'
        return styles

    # NaN (calendar padding) renders as an empty cell rather than "nan".
    styler = (df_enso.style
              .format(lambda v: "" if pd.isna(v) else f"{v:.2f}")
              .apply(style_func, axis=None))
    return styler


def save_enso_table_html(enso_traj, output_path="plots/enso_table.html", threshold=0.5,
                          wandb_enabled=False, start_month=0):
    """
    Save ENSO table as an interactive HTML file.

    Args:
        enso_traj (array): ENSO time series (already ONI)
        output_path (str): Path to save HTML file
        threshold (float): Threshold for coloring
        wandb_enabled (bool): Log to W&B
        start_month (int): 0-based calendar month of the first value (see create_enso_table)

    Returns:
        str: Path to saved file
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Create and style table
    df_enso = create_enso_table(enso_traj, threshold, start_month=start_month)
    styler = style_enso_table(df_enso, threshold)
    
    # Add metadata to HTML
    html_string = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h2 {{ color: #333; }}
            .info {{ background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            table {{ border-collapse: collapse; margin: 20px 0; }}
            td, th {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h2>ENSO Index Table (Year x 3-Month Seasons)</h2>
        <div class="info">
            <p><strong>Color Legend:</strong></p>
            <p><span style="background-color: #FFB3B3; padding: 2px 8px; font-weight: bold;">RED</span> = El Nino (> {threshold})</p>
            <p><span style="background-color: #B3D9FF; padding: 2px 8px; font-weight: bold;">BLUE</span> = La Nina (< -{threshold})</p>
            <p><span style="background-color: #E8E8E8; padding: 2px 8px; font-weight: bold;">GRAY</span> = Neutral</p>
            <p><strong>Data:</strong> {len(enso_traj)} months ({len(enso_traj)/12:.1f} years)</p>
        </div>
        {styler.to_html()}
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_string)
    
    if wandb_enabled and wandb.run is not None:
        wandb.save(output_path)
    
    return output_path


def save_enso_table_matplotlib(enso_traj, output_path="plots/enso_table.png", threshold=0.5,
                               start_month=0):
    """
    Save ENSO table as a PNG image using matplotlib.

    Args:
        enso_traj (array): ENSO time series (already ONI)
        output_path (str): Path to save PNG file
        threshold (float): Threshold for coloring
        start_month (int): 0-based calendar month of the first value (see create_enso_table)

    Returns:
        str: Path to saved file
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Create and style table
    df_enso = create_enso_table(enso_traj, threshold, start_month=start_month)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')

    # Create table. NaN (calendar padding) renders as a blank cell.
    cell_text = np.array([["" if pd.isna(val) else f"{val:.2f}" for val in row]
                          for row in df_enso.values])
    
    table = ax.table(cellText=cell_text, 
                     colLabels=df_enso.columns,
                     rowLabels=df_enso.index,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color the cells, bold only peaks
    peaks = _find_peak_positions(df_enso, threshold)
    for i in range(len(df_enso)):
        for j in range(len(df_enso.columns)):
            val = df_enso.iloc[i, j]
            cell = table[(i+1, j)]
            if pd.isna(val):
                cell.set_facecolor('white')  # calendar padding: blank
                cell.set_text_props(ha='center', va='center')
                continue
            is_peak = (i, j) in peaks
            weight = 'bold' if is_peak else 'normal'

            if val > threshold:
                cell.set_facecolor('#FFB3B3')
                cell.set_text_props(weight=weight, color='red')
            elif val < -threshold:
                cell.set_facecolor('#B3D9FF')
                cell.set_text_props(weight=weight, color='blue')
            else:
                cell.set_facecolor('#E8E8E8')

            cell.set_text_props(ha='center', va='center')
    
    # Style header
    for j in range(len(df_enso.columns)):
        cell = table[(0, j)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Style row labels
    for i in range(len(df_enso)):
        cell = table[(i+1, -1)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Add title and legend
    plt.suptitle('ENSO Index Table (Year × 3-Month Seasons)', fontsize=16, fontweight='bold', y=0.98)
    
    # Add legend
    red_patch = mpatches.Patch(color='#FFB3B3', label=f'El Niño (> {threshold})')
    blue_patch = mpatches.Patch(color='#B3D9FF', label=f'La Niña (< {-threshold})')
    gray_patch = mpatches.Patch(color='#E8E8E8', label='Neutral')
    plt.legend(handles=[red_patch, blue_patch, gray_patch], loc='upper right', bbox_to_anchor=(1.0, 0.02))
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def log_enso_table_wandb(enso_traj, threshold=0.5, start_month=0):
    """
    Log styled ENSO table directly to W&B as rendered HTML.

    Args:
        enso_traj (array): ENSO time series (already ONI)
        threshold (float): Threshold for coloring
        start_month (int): 0-based calendar month of the first value (see create_enso_table)
    """
    if wandb.run is None:
        print("[WARNING] W&B not initialized. Skipping W&B table logging.")
        return

    df_enso = create_enso_table(enso_traj, threshold, start_month=start_month)
    styler = style_enso_table(df_enso, threshold)
    wandb.log({"enso_table": wandb.Html(styler.to_html())})
    print("[OK] Styled ENSO table logged to W&B")
