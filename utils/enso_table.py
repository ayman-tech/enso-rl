"""
ENSO Table Visualization utilities.
Converts ENSO trajectory into styled year-month table with colors.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb


def create_enso_table(enso_traj, threshold=0.5):
    """
    Create a formatted ENSO table (years x 3-month seasons).
    
    Args:
        enso_traj (array): ENSO time series (already smooth / ONI)
        threshold (float): Threshold for coloring
        
    Returns:
        pd.DataFrame: Styled dataframe with ENSO values
    """
    enso_data = np.asarray(enso_traj)
    
    # Reshape into years x seasons
    n_months = 12
    n_years = len(enso_data) // n_months
    data_full_years = enso_data[:n_years * n_months]
    reshaped_data = data_full_years.reshape(n_years, n_months)
    
    # Create DataFrame with 3-month season labels
    months = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
    df_enso = pd.DataFrame(reshaped_data, columns=months)
    df_enso.index.name = 'Year'
    df_enso.index = df_enso.index + 1
    
    return df_enso


def style_enso_table(df_enso, threshold=0.5):
    """
    Apply styling to ENSO table.
    
    Args:
        df_enso (pd.DataFrame): ENSO dataframe
        threshold (float): Threshold for coloring
        
    Returns:
        pd.io.formats.style.Styler: Styled dataframe
    """
    def color_cells(val):
        """Color cells based on ENSO value."""
        if val > threshold:
            return 'background-color: #FFB3B3; color: red; font-weight: bold'  # Light red
        elif val < -threshold:
            return 'background-color: #B3D9FF; color: blue; font-weight: bold'  # Light blue
        else:
            return 'background-color: #E8E8E8; color: black'  # Light gray
    
    styler = df_enso.style.format("{:.2f}").map(color_cells)
    return styler


def save_enso_table_html(enso_traj, output_path="plots/enso_table.html", threshold=0.5, 
                          wandb_enabled=False):
    """
    Save ENSO table as an interactive HTML file.
    
    Args:
        enso_traj (array): ENSO time series (already ONI)
        output_path (str): Path to save HTML file
        threshold (float): Threshold for coloring
        wandb_enabled (bool): Log to W&B
        
    Returns:
        str: Path to saved file
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Create and style table
    df_enso = create_enso_table(enso_traj, threshold)
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
                               wandb_enabled=False):
    """
    Save ENSO table as a PNG image using matplotlib.
    
    Args:
        enso_traj (array): ENSO time series (already ONI)
        output_path (str): Path to save PNG file
        threshold (float): Threshold for coloring
        wandb_enabled (bool): Log to W&B
        
    Returns:
        str: Path to saved file
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Create and style table
    df_enso = create_enso_table(enso_traj, threshold)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    # Format cell values to max 2 decimal places
    cell_text = np.array([[f"{val:.2f}" for val in row] for row in df_enso.values])
    
    table = ax.table(cellText=cell_text, 
                     colLabels=df_enso.columns,
                     rowLabels=df_enso.index,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color the cells
    for i in range(len(df_enso)):
        for j in range(len(df_enso.columns)):
            val = df_enso.iloc[i, j]
            cell = table[(i+1, j)]
            
            if val > threshold:
                cell.set_facecolor('#FFB3B3')  # Light red
                cell.set_text_props(weight='bold', color='red')
            elif val < -threshold:
                cell.set_facecolor('#B3D9FF')  # Light blue
                cell.set_text_props(weight='bold', color='blue')
            else:
                cell.set_facecolor('#E8E8E8')  # Light gray
            
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
    
    if wandb_enabled and wandb.run is not None:
        wandb.log({f"enso_table_image": wandb.Image(output_path)})
    
    return output_path


def log_enso_table_wandb(enso_traj, threshold=0.5):
    """
    Log ENSO table directly to W&B as a formatted table.
    
    Args:
        enso_traj (array): ENSO time series (already ONI)
        threshold (float): Threshold for coloring
    """
    if wandb.run is None:
        print("[WARNING] W&B not initialized. Skipping W&B table logging.")
        return
    
    df_enso = create_enso_table(enso_traj, threshold)
    
    # Create W&B table
    table_data = []
    for year, row in df_enso.iterrows():
        row_data = [year] + [f"{val:.2f}" for val in row.values]
        table_data.append(row_data)
    
    columns = ['Year'] + list(df_enso.columns)
    wandb_table = wandb.Table(data=table_data, columns=columns)
    
    wandb.log({"enso_table": wandb_table})
    print("[OK] ENSO table logged to W&B")
