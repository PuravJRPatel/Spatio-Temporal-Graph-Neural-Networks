import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse
from matplotlib.dates import YearLocator
from matplotlib import cm
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

def extract_temperature_rainfall_data(embeddings, temp_indices, rainfall_indices):
    """
    Extract temperature and rainfall data from embeddings.
    
    Args:
        embeddings: Dictionary of embeddings by year
        temp_indices: Indices of temperature data within embeddings
        rainfall_indices: Indices of rainfall data within embeddings
        
    Returns:
        Tuple of dictionaries: (temperature_data, rainfall_data)
    """
    years = sorted(embeddings.keys())
    
    # If indices aren't specified, try to infer based on typical patterns
    if temp_indices is None:
        # For demonstration, use first element - in real use, would analyze patterns
        temp_indices = [0]
    
    if rainfall_indices is None:
        # For demonstration, use second element - in real use, would analyze patterns
        rainfall_indices = [1]
        
    # Extract data
    temperature_data = {}
    rainfall_data = {}
    
    for year in years:
        embedding = embeddings[year]
        
        # Average temperature features if multiple
        if len(temp_indices) > 0:
            temperature_data[year] = torch.mean(embedding[temp_indices]).item()
        
        # Average rainfall features if multiple
        if len(rainfall_indices) > 0:
            rainfall_data[year] = torch.mean(embedding[rainfall_indices]).item()
            
    return temperature_data, rainfall_data

def normalize_data(data_dict):
    """
    Normalize data to a reasonable scale if needed
    """
    values = list(data_dict.values())
    min_val = min(values)
    max_val = max(values)
    
    # Only normalize if the range is very large or very small
    if max_val - min_val > 100 or max_val - min_val < 0.1:
        normalized = {}
        for year, value in data_dict.items():
            normalized[year] = (value - min_val) / (max_val - min_val) * 10
        return normalized
    
    return data_dict

def visualize_climate_predictions(historical_embeddings, future_embeddings, 
                                 climate_mode="standard", output_dir="./plots",
                                 temp_indices=None, rainfall_indices=None):
    """
    Create visualizations for climate predictions
    
    Args:
        historical_embeddings: Dictionary of historical embeddings
        future_embeddings: Dictionary of predicted future embeddings
        climate_mode: The climate scenario used for prediction
        output_dir: Directory to save plots
        temp_indices: Indices for temperature data
        rainfall_indices: Indices for rainfall data
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Combine historical and future embeddings
    all_embeddings = {**historical_embeddings, **future_embeddings}
    
    # Extract temperature and rainfall data
    temp_data, rainfall_data = extract_temperature_rainfall_data(
        all_embeddings, temp_indices, rainfall_indices)
    
    # Split into historical and future periods
    historical_years = sorted([y for y in historical_embeddings.keys()])
    all_years = sorted(all_embeddings.keys())
    future_years = [y for y in all_years if y > max(historical_years)]
    future_years_boundary = min(future_years) if future_years else max(historical_years) + 1
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Color schemes
    if climate_mode == "warming":
        cmap = "YlOrRd"
        title_addon = "- Warming Scenario"
    elif climate_mode == "extreme":
        cmap = "plasma"
        title_addon = "- Extreme Scenario"
    else:
        cmap = "viridis"
        title_addon = "- Standard Scenario"
    
    # Set the main title
    fig.suptitle(f"Climate Predictions {title_addon}", fontsize=16)
    
    # Plot temperature data
    plot_climate_variable(ax1, temp_data, historical_years, future_years, 
                         "Temperature", "°C", climate_mode, cmap)
    
    # Plot rainfall data
    plot_climate_variable(ax2, rainfall_data, historical_years, future_years, 
                         "Precipitation", "mm", climate_mode, cmap)
    
    # Add a vertical line separating historical and predicted data
    for ax in [ax1, ax2]:
        ax.axvline(x=future_years_boundary-0.5, color='gray', linestyle='--', alpha=0.7)
        ax.text(future_years_boundary, ax.get_ylim()[1]*0.95, 
                "Predictions →", ha='left', va='top', alpha=0.7)
    
    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save the figure
    output_file = os.path.join(output_dir, f"climate_prediction_{climate_mode}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    # Show the figure
    plt.show()
    
    # Additional visualizations
    create_climate_heatmap(all_embeddings, historical_years, future_years, 
                          output_dir, climate_mode)
    
    create_trend_visualization(temp_data, rainfall_data, historical_years, 
                              future_years, output_dir, climate_mode)
    
def plot_climate_variable(ax, data, historical_years, future_years, 
                         variable_name, unit, climate_mode, cmap_name):
    """
    Plot a single climate variable on the given axis
    """
    # Get all years
    all_years = sorted(list(data.keys()))
    values = [data[year] for year in all_years]
    
    # Historical data
    hist_years = [y for y in all_years if y in historical_years]
    hist_values = [data[y] for y in hist_years]
    
    # Future data
    fut_years = [y for y in all_years if y in future_years]
    fut_values = [data[y] for y in fut_years]
    
    # Set up colormap for future data based on climate mode
    if climate_mode == "warming":
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(fut_years)))
    elif climate_mode == "extreme":
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(fut_years)))
    else:
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fut_years)))
        
    # Plot historical data
    ax.plot(hist_years, hist_values, marker='o', linestyle='-', color='blue',
           label='Historical Data')
    
    # Plot future data line
    if fut_years:
        ax.plot(fut_years, fut_values, linestyle='--', color='red',
               label='Predicted Trend')
        
        # Plot colored points for future data to show intensity
        for i, (year, value) in enumerate(zip(fut_years, fut_values)):
            ax.scatter(year, value, color=colors[i], s=50, zorder=5)
    
    # Add confidence band for future predictions
    if fut_years and climate_mode == "extreme":
        # For extreme scenario, show wider uncertainty
        std_dev = np.std(hist_values) * 1.5
        ax.fill_between(fut_years, 
                        [v - std_dev for v in fut_values],
                        [v + std_dev for v in fut_values],
                        color='red', alpha=0.1)
    elif fut_years:
        # For other scenarios, show standard uncertainty
        std_dev = np.std(hist_values)
        ax.fill_between(fut_years, 
                        [v - std_dev for v in fut_values],
                        [v + std_dev for v in fut_values],
                        color='red', alpha=0.1)
    
    # Set labels and title
    ax.set_ylabel(f"{variable_name} ({unit})")
    ax.set_title(f"{variable_name} Over Time")
    
    # Format x-axis to show years
    ax.xaxis.set_major_locator(YearLocator(5))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    # Add climate scenario indicator
    if climate_mode == "warming":
        ax.text(0.02, 0.95, "Warming Trend Scenario", transform=ax.transAxes,
               fontsize=10, bbox=dict(facecolor='red', alpha=0.1))
    elif climate_mode == "extreme":
        ax.text(0.02, 0.95, "Extreme Variability Scenario", transform=ax.transAxes,
               fontsize=10, bbox=dict(facecolor='purple', alpha=0.1))
    
def create_climate_heatmap(embeddings, historical_years, future_years, 
                         output_dir, climate_mode):
    """
    Create a heatmap showing the evolution of all embedding dimensions over time
    """
    all_years = sorted(list(embeddings.keys()))
    
    # Get the shape of embeddings
    sample_embedding = embeddings[all_years[0]]
    if isinstance(sample_embedding, torch.Tensor):
        num_features = sample_embedding.shape[0]
    else:
        num_features = len(sample_embedding)
    
    # Extract data
    data = np.zeros((len(all_years), num_features))
    for i, year in enumerate(all_years):
        embedding = embeddings[year]
        if isinstance(embedding, torch.Tensor):
            data[i, :] = embedding.cpu().numpy()
        else:
            data[i, :] = np.array(embedding)
    
    # Normalize data for better visualization
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    norm_data = (data - data_mean) / (data_std + 1e-8)
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Determine color scheme based on climate mode
    if climate_mode == "warming":
        cmap = LinearSegmentedColormap.from_list("", ["white", "yellow", "red", "darkred"])
    elif climate_mode == "extreme":
        cmap = "RdBu_r"
    else:
        cmap = "viridis"
    
    # Create heatmap
    plt.imshow(norm_data.T, aspect='auto', cmap=cmap, 
              interpolation='nearest', vmin=-2, vmax=2)
    
    # Add vertical line separating historical and future
    future_boundary = all_years.index(min(future_years)) if future_years else len(all_years)
    plt.axvline(x=future_boundary-0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Set labels and title
    plt.colorbar(label='Normalized Value')
    plt.xlabel('Year')
    plt.ylabel('Feature Dimension')
    plt.title(f'Climate Data Evolution Heatmap - {climate_mode.capitalize()} Scenario')
    
    # Set x-ticks to years (but not all of them to avoid crowding)
    tick_indices = np.linspace(0, len(all_years)-1, 10, dtype=int)
    plt.xticks(tick_indices, [all_years[i] for i in tick_indices])
    
    # Save the figure
    output_file = os.path.join(output_dir, f"climate_heatmap_{climate_mode}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")
    
def create_trend_visualization(temp_data, rainfall_data, historical_years, 
                             future_years, output_dir, climate_mode):
    """
    Create a trend visualization showing the relationship between temperature and rainfall
    """
    # Convert data to lists aligned by year
    all_years = sorted(list(set(temp_data.keys()) & set(rainfall_data.keys())))
    historical_all = [y for y in all_years if y in historical_years]
    future_all = [y for y in all_years if y in future_years]
    
    # Get temperature and rainfall values
    hist_temp = [temp_data[y] for y in historical_all]
    hist_rain = [rainfall_data[y] for y in historical_all]
    
    fut_temp = [temp_data[y] for y in future_all]
    fut_rain = [rainfall_data[y] for y in future_all]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot historical data
    sc1 = plt.scatter(hist_temp, hist_rain, c=historical_all, cmap='Blues', 
                    s=60, label='Historical Data')
    
    # Plot future data with appropriate color scheme
    if climate_mode == "warming":
        cmap = "YlOrRd"
    elif climate_mode == "extreme":
        cmap = "plasma"
    else:
        cmap = "viridis"
        
    if future_all:
        sc2 = plt.scatter(fut_temp, fut_rain, c=future_all, cmap=cmap, 
                        s=80, marker='s', label='Predicted Data')
    
        # Add arrows to show progression
        all_temp = hist_temp + fut_temp
        all_rain = hist_rain + fut_rain
        all_years_list = historical_all + future_all
        
        # Plot arrows between consecutive points
        for i in range(1, len(all_years_list)):
            # Make arrows for future predictions more prominent
            if all_years_list[i] in future_all:
                arrow_color = 'red'
                arrow_width = 1.2
            else:
                arrow_color = 'blue'
                arrow_width = 0.8
                
            plt.annotate('', 
                        xy=(all_temp[i], all_rain[i]),
                        xytext=(all_temp[i-1], all_rain[i-1]),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, 
                                       lw=arrow_width, alpha=0.7))
    
    # Labels and title
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Precipitation (mm)')
    plt.title(f'Temperature-Precipitation Relationship - {climate_mode.capitalize()} Scenario')
    
    # Add colorbar for years
    plt.colorbar(sc1, label='Historical Years')
    if future_all:
        plt.colorbar(sc2, label='Future Years')
    
    # Add grid
    plt.grid(alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Save figure
    output_file = os.path.join(output_dir, f"climate_trends_{climate_mode}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Trend visualization saved to {output_file}")

def calculate_climate_metrics(historical_embeddings, future_embeddings, 
                            climate_mode, temp_indices, rainfall_indices):
    """
    Calculate and display key climate metrics
    """
    # Extract temperature and rainfall data
    all_embeddings = {**historical_embeddings, **future_embeddings}
    temp_data, rainfall_data = extract_temperature_rainfall_data(
        all_embeddings, temp_indices, rainfall_indices)
    
    # Split into historical and future
    historical_years = sorted(historical_embeddings.keys())
    all_years = sorted(all_embeddings.keys())
    future_years = [y for y in all_years if y > max(historical_years)]
    
    # Calculate metrics
    hist_temp = [temp_data[y] for y in historical_years]
    fut_temp = [temp_data[y] for y in future_years]
    
    hist_rain = [rainfall_data[y] for y in historical_years]
    fut_rain = [rainfall_data[y] for y in future_years]
    
    # Create metrics dictionary
    metrics = {
        "Average Temperature (Historical)": np.mean(hist_temp),
        "Average Temperature (Predicted)": np.mean(fut_temp) if fut_temp else None,
        "Temperature Change": np.mean(fut_temp) - np.mean(hist_temp) if fut_temp else None,
        "Temperature Trend (Historical)": (hist_temp[-1] - hist_temp[0]) / len(hist_temp),
        "Temperature Trend (Predicted)": (fut_temp[-1] - fut_temp[0]) / len(fut_temp) if fut_temp and len(fut_temp) > 1 else None,
        
        "Average Precipitation (Historical)": np.mean(hist_rain),
        "Average Precipitation (Predicted)": np.mean(fut_rain) if fut_rain else None,
        "Precipitation Change": np.mean(fut_rain) - np.mean(hist_rain) if fut_rain else None,
        "Precipitation Trend (Historical)": (hist_rain[-1] - hist_rain[0]) / len(hist_rain),
        "Precipitation Trend (Predicted)": (fut_rain[-1] - fut_rain[0]) / len(fut_rain) if fut_rain and len(fut_rain) > 1 else None,
        
        "Temperature Variability (Historical)": np.std(hist_temp),
        "Temperature Variability (Predicted)": np.std(fut_temp) if fut_temp else None,
        "Precipitation Variability (Historical)": np.std(hist_rain),
        "Precipitation Variability (Predicted)": np.std(fut_rain) if fut_rain else None,
    }
    
    # Create DataFrame for nicer display
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_df['Value'] = metrics_df['Value'].round(4)
    
    # Save metrics
    output_file = os.path.join(output_dir, f"climate_metrics_{climate_mode}.csv")
    metrics_df.to_csv(output_file, index=False)
    print(f"Climate metrics saved to {output_file}")
    
    return metrics_df

def main():
    parser = argparse.ArgumentParser(description='Climate Prediction Visualization')
    parser.add_argument('--embeddings_dir', type=str, default='backend/data/graph_embeddings',
                      help='Directory containing embedding files')
    parser.add_argument('--output_dir', type=str, default='./plots',
                      help='Directory to save visualization outputs')
    parser.add_argument('--climate_mode', type=str, default='warming',
                      choices=['standard', 'warming', 'extreme'],
                      help='Climate scenario mode')
    parser.add_argument('--temp_indices', type=str, default=None,
                      help='Comma-separated list of temperature indices')
    parser.add_argument('--rainfall_indices', type=str, default=None,
                      help='Comma-separated list of rainfall indices')
    
    args = parser.parse_args()
    
    # Process indices if provided
    if args.temp_indices:
        temp_indices = [int(x) for x in args.temp_indices.split(',')]
    else:
        temp_indices = None
        
    if args.rainfall_indices:
        rainfall_indices = [int(x) for x in args.rainfall_indices.split(',')]
    else:
        rainfall_indices = None
    
    # Load historical embeddings
    historical_embeddings = {}
    future_embeddings = {}
    
    # Check if directory exists
    if not os.path.exists(args.embeddings_dir):
        print(f"Error: Embeddings directory {args.embeddings_dir} does not exist")
        return
    
    # Load historical embeddings
    for year in range(1980, 2025):
        embedding_file = os.path.join(args.embeddings_dir, f"embeddings_{year}.pt")
        if os.path.exists(embedding_file):
            try:
                historical_embeddings[year] = torch.load(embedding_file)
            except Exception as e:
                print(f"Error loading embeddings for year {year}: {e}")
    
    # Look for predicted embeddings
    latest_historical_year = max(historical_embeddings.keys())
    for year in range(latest_historical_year + 1, 2040):
        # First try climate-specific file
        embedding_file = os.path.join(args.embeddings_dir, f"embeddings_{year}_{args.climate_mode}.pt")
        if not os.path.exists(embedding_file):
            # If not found, try generic file
            embedding_file = os.path.join(args.embeddings_dir, f"embeddings_{year}.pt")
            
        if os.path.exists(embedding_file):
            try:
                future_embeddings[year] = torch.load(embedding_file)
            except Exception as e:
                print(f"Error loading predicted embeddings for year {year}: {e}")
    
    if not historical_embeddings:
        print("Error: No historical embeddings found")
        return
        
    if not future_embeddings:
        print("Warning: No future predictions found. Only visualizing historical data.")
    
    # Create visualizations
    visualize_climate_predictions(
        historical_embeddings, 
        future_embeddings,
        args.climate_mode,
        args.output_dir,
        temp_indices,
        rainfall_indices
    )
    
    # Calculate metrics
    metrics = calculate_climate_metrics(
        historical_embeddings,
        future_embeddings,
        args.climate_mode,
        temp_indices,
        rainfall_indices
    )
    
    print("\nClimate Metrics Summary:")
    print(metrics)

if __name__ == "__main__":
    main()