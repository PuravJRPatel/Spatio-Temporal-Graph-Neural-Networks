import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from matplotlib.colors import LinearSegmentedColormap

# Set page configuration
st.set_page_config(
    page_title="India Climate Visualization",
    page_icon="🌡️",
    layout="wide"
)

# Define the base paths based on the project structure from the image
BASE_PATHS = {
    "embeddings": ["backend/data/graph_embeddings", "graph_embeddings"],
    "climate_data": ["backend/data/Processed Data/climate_data.csv", 
                     "backend/data/graph_embeddings/Processed Data/climate_data.csv",
                     "graph_embeddings/Processed Data/climate_data.csv"]
}

# Helper function to find file with dynamic path resolution
def find_file(base_paths, filename=None):
    for base_path in base_paths:
        full_path = os.path.join(base_path, filename) if filename else base_path
        if os.path.exists(full_path):
            return full_path
    return None

# Helper function to load embeddings with better error handling
@st.cache_data
def load_embedding(year):
    embedding_filename = f"embeddings_{year}.pt"
    
    # Find the embedding file
    embedding_path = find_file(BASE_PATHS["embeddings"], embedding_filename)
    
    if embedding_path:
        try:
            embedding = torch.load(embedding_path)
            return embedding
        except Exception as e:
            st.sidebar.error(f"Error loading embedding: {str(e)}")
            return None
    else:
        if year > 2023:
            # Don't show error for future years as we'll use synthetic data
            st.sidebar.info(f"No embedding file found for year {year}. Will use synthetic projection.")
        else:
            st.sidebar.warning(f"Embedding file for year {year} not found.")
        return None

# Load climate data with better error handling
@st.cache_data
def load_climate_data():
    # Try to find the climate data file
    climate_data_path = None
    for path in BASE_PATHS["climate_data"]:
        if os.path.exists(path):
            climate_data_path = path
            break
    
    if climate_data_path:
        try:
            climate_df = pd.read_csv(climate_data_path)
            # Extract unique lat-lon pairs
            coords_df = climate_df[['latitude', 'longitude']].drop_duplicates()
            return coords_df, climate_df
        except Exception as e:
            st.error(f"Error loading climate data: {str(e)}")
            return create_dummy_data()
    else:
        st.warning("Climate data file not found. Creating synthetic data for demonstration.")
        return create_dummy_data()

# Create dummy data if needed
def create_dummy_data():
    years = list(range(1980, 2024))
    num_points = 100
    
    # Generate random coordinates for India
    np.random.seed(42)
    lats = np.random.uniform(8, 35, num_points)
    lons = np.random.uniform(70, 95, num_points)
    
    data_list = []
    for year in years:
        temp_base = 25 + 5 * np.sin((year - 1980) / 10)
        rain_base = 100 + 50 * np.cos((year - 1980) / 8)
        
        for i in range(num_points):
            temp = temp_base + 3 * np.sin(lats[i]/10) + 2 * np.cos(lons[i]/10)
            rain = rain_base + 20 * np.cos(lats[i]/8) + 15 * np.sin(lons[i]/12)
            
            data_list.append({
                'year': year,
                'latitude': lats[i],
                'longitude': lons[i],
                'temperature': temp,
                'rainfall': rain
            })
    
    climate_df = pd.DataFrame(data_list)
    coords_df = climate_df[['latitude', 'longitude']].drop_duplicates()
    
    return coords_df, climate_df

# Process embedding data more efficiently
def process_climate_data(embedding, coordinates_df, climate_df, year):
    # Get coordinates
    lats = coordinates_df['latitude'].values
    lons = coordinates_df['longitude'].values
    
    # For historical years (<=2023), try to use actual data
    if year <= 2023:
        year_data = climate_df[climate_df['year'] == year]
        if not year_data.empty:
            # Match coordinates order with the data
            rainfall = []
            temperature = []
            for _, coord in coordinates_df.iterrows():
                matching_rows = year_data[
                    (year_data['latitude'] == coord['latitude']) & 
                    (year_data['longitude'] == coord['longitude'])
                ]
                if not matching_rows.empty:
                    rainfall.append(matching_rows['rainfall'].values[0])
                    temperature.append(matching_rows['temperature'].values[0])
                else:
                    # Use mean values if exact match not found
                    rainfall.append(year_data['rainfall'].mean())
                    temperature.append(year_data['temperature'].mean())
            
            return {
                'lats': lats,
                'lons': lons,
                'rainfall': np.array(rainfall),
                'temperature': np.array(temperature)
            }
    
    # For prediction years or if historical data not found
    if embedding is not None:
        # Extract from embedding
        rainfall, temperature = extract_from_embedding(embedding, coordinates_df, climate_df)
    else:
        # Create synthetic projection
        rainfall, temperature = create_synthetic_projection(coordinates_df, climate_df, year)
    
    return {
        'lats': lats,
        'lons': lons,
        'rainfall': rainfall,
        'temperature': temperature
    }

# Create synthetic projections for years without embeddings
def create_synthetic_projection(coordinates_df, climate_df, year):
    # Find latest available year data
    latest_year = climate_df['year'].max()
    early_year = max(climate_df['year'].min(), latest_year - 5)
    
    # Get average rates of change
    latest_temp = climate_df[climate_df['year'] == latest_year]['temperature'].mean()
    early_temp = climate_df[climate_df['year'] == early_year]['temperature'].mean()
    temp_rate = (latest_temp - early_temp) / (latest_year - early_year)
    
    latest_rain = climate_df[climate_df['year'] == latest_year]['rainfall'].mean()
    early_rain = climate_df[climate_df['year'] == early_year]['rainfall'].mean()
    rain_rate = (latest_rain - early_rain) / (latest_year - early_year)
    
    # Project forward
    years_forward = year - latest_year
    
    # Get base data for the latest year that matches our coordinates
    base_temp = []
    base_rain = []
    
    latest_data = climate_df[climate_df['year'] == latest_year]
    for _, coord in coordinates_df.iterrows():
        matching_rows = latest_data[
            (latest_data['latitude'] == coord['latitude']) & 
            (latest_data['longitude'] == coord['longitude'])
        ]
        if not matching_rows.empty:
            base_temp.append(matching_rows['temperature'].values[0])
            base_rain.append(matching_rows['rainfall'].values[0])
        else:
            # Use mean if exact match not found
            base_temp.append(latest_temp)
            base_rain.append(latest_rain)
    
    # Apply projection with some randomness
    np.random.seed(year)  # Consistent randomness per year
    projected_temp = np.array(base_temp) + (temp_rate * years_forward) + np.random.normal(0, 0.2, len(base_temp))
    projected_rain = np.array(base_rain) + (rain_rate * years_forward) + np.random.normal(0, 1.0, len(base_rain))
    projected_rain = np.maximum(projected_rain, 0.1)  # Ensure rainfall isn't negative
    
    return projected_rain, projected_temp

# Extract data from embedding
def extract_from_embedding(embedding, coordinates_df, climate_df):
    # Get historical min/max for normalization
    temp_min, temp_max = 0, 40
    rain_min, rain_max = 0, 46
    
    # Process embedding based on its type
    features = None
    
    if isinstance(embedding, dict):
        # Try common key names in different models
        for key in ['node_features', 'embeddings', 'x']:
            if key in embedding:
                features = embedding[key]
                break
        
        # If no known keys, try first value
        if features is None and len(embedding) > 0:
            features = list(embedding.values())[0]
    
    elif isinstance(embedding, torch.Tensor):
        features = embedding
    
    # Convert tensor to numpy if needed
    if isinstance(features, torch.Tensor):
        features = features.cpu().detach().numpy()
    
    # Handle different shapes
    if features is None:
        # Return synthetic data if we can't process the embedding
        return create_synthetic_projection(coordinates_df, climate_df, 2024)
    
    if len(features.shape) == 3:  # [batch, nodes, features]
        features = features[0]  # Take first batch
    
    # Match feature count to coordinate count
    if features.shape[0] != len(coordinates_df):
        # Truncate or pad as needed
        if features.shape[0] > len(coordinates_df):
            features = features[:len(coordinates_df)]
        else:
            features = np.pad(features, ((0, len(coordinates_df) - features.shape[0]), (0, 0)))
    
    # Extract rainfall and temperature from features
    if features.shape[1] >= 2:
        rainfall_embedding = features[:, 0]
        temperature_embedding = features[:, 1]
    else:
        rainfall_embedding = features[:, 0]
        temperature_embedding = features[:, 0] * 1.2  # Add variation
    
    # Normalize to realistic ranges
    rainfall = rain_min + (rain_max - rain_min) * (rainfall_embedding - np.min(rainfall_embedding)) / (np.max(rainfall_embedding) - np.min(rainfall_embedding) + 1e-10)
    temperature = temp_min + (temp_max - temp_min) * (temperature_embedding - np.min(temperature_embedding)) / (np.max(temperature_embedding) - np.min(temperature_embedding) + 1e-10)
    
    return rainfall, temperature

# Plot heatmap
def plot_heatmap(data, metric, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    values = data[metric]
    
    if metric == 'rainfall':
        cmap = LinearSegmentedColormap.from_list('rainfall_cmap', ['#f7fbff', '#08306b'])
        vmin, vmax = np.min(values) * 0.9, np.max(values) * 1.1
        cb_label = 'Rainfall (mm)'
    else:
        cmap = LinearSegmentedColormap.from_list('temp_cmap', ['#ffffcc', '#e31a1c'])
        vmin, vmax = np.min(values) * 0.9, np.max(values) * 1.1
        cb_label = 'Temperature (°C)'
    
    scatter = ax.scatter(
        data['lons'], 
        data['lats'], 
        c=values, 
        cmap=cmap,
        alpha=0.8,
        s=50,
        vmin=vmin,
        vmax=vmax
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(cb_label)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Set limits for India's geography
    ax.set_xlim(68, 98)
    ax.set_ylim(6, 38)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return fig

# Main app
def main():
    st.title("🌡️ India Climate Visualization (1980-2033)")
    st.write("Explore temperature and rainfall patterns across India with historical data and predictions.")
    
    # File system exploration section (only in debug mode)
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    if debug_mode:
        st.sidebar.header("File System Exploration")
        
        # Allow manual path exploration
        base_path = st.sidebar.text_input("Enter base path to explore", "backend/data")
        
        if os.path.exists(base_path):
            st.sidebar.success(f"Path exists: {base_path}")
            try:
                files = os.listdir(base_path)
                st.sidebar.write("Contents:")
                for f in files:
                    full_path = os.path.join(base_path, f)
                    is_dir = os.path.isdir(full_path)
                    st.sidebar.write(f"{'📁' if is_dir else '📄'} {f}")
            except Exception as e:
                st.sidebar.error(f"Error listing directory: {str(e)}")
        else:
            st.sidebar.error(f"Path does not exist: {base_path}")
    
    # Load data
    coordinates_df, climate_df = load_climate_data()
    
    if coordinates_df is not None and climate_df is not None:
        # Get available years
        historical_years = sorted(climate_df['year'].unique())
        prediction_years = list(range(max(historical_years) + 1, 2034))
        available_years = historical_years + prediction_years
        
        # Year selection
        st.sidebar.header("Settings")
        selected_year = st.sidebar.slider(
            "Select Year",
            min_value=min(available_years),
            max_value=max(prediction_years),
            value=2023,
            step=1
        )
        
        # Load embedding and process data
        embedding = load_embedding(selected_year)
        heatmap_data = process_climate_data(embedding, coordinates_df, climate_df, selected_year)
        
        # Display year info
        st.header(f"Climate Data for {selected_year}")
        if selected_year > max(historical_years):
            st.info("⚠️ Data shown is predicted based on graph embeddings or projection models.")
        else:
            st.info("📊 Historical data from records.")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Temperature Map")
            temp_fig = plot_heatmap(heatmap_data, 'temperature', f"Temperature Distribution ({selected_year})")
            st.pyplot(temp_fig)
        
        with col2:
            st.subheader("Rainfall Map")
            rain_fig = plot_heatmap(heatmap_data, 'rainfall', f"Rainfall Distribution ({selected_year})")
            st.pyplot(rain_fig)
        
        # Statistics
        st.subheader("Summary Statistics")
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            mean_temp = np.mean(heatmap_data['temperature'])
            baseline_year = min(2023, max(historical_years))
            baseline_temp_data = climate_df[climate_df['year'] == baseline_year]['temperature']
            
            if not baseline_temp_data.empty:
                baseline_temp = baseline_temp_data.mean()
                
                st.metric(
                    "Average Temperature", 
                    f"{mean_temp:.2f}°C",
                    f"{mean_temp - baseline_temp:.2f}°C vs. {baseline_year}" 
                    if selected_year != baseline_year else None
                )
            else:
                st.metric("Average Temperature", f"{mean_temp:.2f}°C")
        
        with stats_col2:
            mean_rain = np.mean(heatmap_data['rainfall'])
            baseline_rain_data = climate_df[climate_df['year'] == baseline_year]['rainfall']
            
            if not baseline_rain_data.empty:
                baseline_rain = baseline_rain_data.mean()
                
                st.metric(
                    "Average Rainfall", 
                    f"{mean_rain:.2f} mm",
                    f"{((mean_rain / baseline_rain) - 1) * 100:.1f}%" 
                    if selected_year != baseline_year else None
                )
            else:
                st.metric("Average Rainfall", f"{mean_rain:.2f} mm")
        
        # Data explorer
        with st.expander("Show Raw Data"):
            display_data = pd.DataFrame({
                'Latitude': heatmap_data['lats'],
                'Longitude': heatmap_data['lons'],
                'Temperature (°C)': heatmap_data['temperature'],
                'Rainfall (mm)': heatmap_data['rainfall']
            })
            
            st.dataframe(display_data)
    
    else:
        st.error("Could not load climate data. Please check file structure and paths.")

if __name__ == "__main__":
    main()