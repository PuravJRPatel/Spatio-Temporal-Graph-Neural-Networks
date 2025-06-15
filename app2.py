import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.patches as mpatches

# Set page configuration
st.set_page_config(
    page_title="India Climate Visualization",
    page_icon="🌡️",
    layout="wide"
)

BASE_PATHS = {
    "embeddings": ["backend/data/graph_embeddings", "graph_embeddings"],
    "climate_data": ["backend/data/Processed Data/climate_data.csv", 
                     "backend/data/graph_embeddings/Processed Data/climate_data.csv",
                     "graph_embeddings/Processed Data/climate_data.csv"],
    "shapefiles": ["backend/data/India Shape", "shapefiles", "data/India Shape"]
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

# Load India state boundaries
@st.cache_data
def load_india_states():
    # Try to find shapefile in various locations
    shapefile_path = None
    shapefile_names = ["india_st.shp", "india_admin.shp", "india.shp"]
    
    for base_path in BASE_PATHS["shapefiles"]:
        for name in shapefile_names:
            path = os.path.join(base_path, name)
            if os.path.exists(path):
                shapefile_path = path
                break
        if shapefile_path:
            break
    
    if shapefile_path:
        try:
            india_states = gpd.read_file(shapefile_path)
            return india_states
        except Exception as e:
            st.warning(f"Error loading shapefile: {str(e)}")
    
    # Create dummy state boundaries if shapefile not found
    return create_dummy_states()

# Create dummy India state boundaries if shapefile not found
def create_dummy_states():
    
    dummy_states = [
        ["Maharashtra", [72, 16, 80, 22]],
        ["Gujarat", [68, 20, 74, 24]],
        ["Tamil Nadu", [76, 8, 80, 13]],
        ["Kerala", [75, 8, 77, 13]],
        ["Karnataka", [74, 12, 78, 18]],
        ["Andhra Pradesh", [77, 13, 84, 19]],
        ["Telangana", [77, 16, 81, 20]],
        ["Madhya Pradesh", [74, 21, 83, 26]],
        ["Rajasthan", [69, 23, 78, 30]],
        ["Uttar Pradesh", [77, 24, 84, 30]],
        ["Bihar", [83, 24, 88, 28]],
        ["West Bengal", [86, 21, 90, 27]],
        ["Odisha", [82, 17, 87, 22]],
        ["Jharkhand", [83, 22, 87, 25]],
        ["Chhattisgarh", [80, 17, 84, 24]],
        ["Punjab", [73, 29, 77, 33]],
        ["Haryana", [74, 27, 78, 31]],
        ["Uttarakhand", [77, 29, 81, 32]],
        ["Himachal Pradesh", [75, 30, 79, 33]],
        ["Jammu and Kashmir", [73, 32, 80, 37]],
        ["Assam", [89, 24, 96, 28]],
    ]
    
    # Create a GeoDataFrame from dummy data
    geometries = []
    names = []
    for state in dummy_states:
        name = state[0]
        min_lon, min_lat, max_lon, max_lat = state[1]
        # Create a simple polygon from the bounding box
        from shapely.geometry import Polygon
        poly = Polygon([
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat)
        ])
        geometries.append(poly)
        names.append(name)
    
    states_gdf = gpd.GeoDataFrame({"state_name": names}, geometry=geometries)
    states_gdf.crs = "EPSG:4326"  # WGS84 coordinate system
    
    return states_gdf

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
    temp_min, temp_max = 21, 34
    rain_min, rain_max = 0, 20
    
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
        temperature_embedding = features[:, 0] * 1.2  
    

    rainfall = rain_min + (rain_max - rain_min) * (rainfall_embedding - np.min(rainfall_embedding)) / (np.max(rainfall_embedding) - np.min(rainfall_embedding) + 1e-10)
    temperature = temp_min + (temp_max - temp_min) * (temperature_embedding - np.min(temperature_embedding)) / (np.max(temperature_embedding) - np.min(temperature_embedding) + 1e-10)
    
    return rainfall, temperature


def calculate_regional_averages(heatmap_data, india_states):
    # Create points from lat/lon
    points = [Point(lon, lat) for lon, lat in zip(heatmap_data['lons'], heatmap_data['lats'])]
    points_gdf = gpd.GeoDataFrame(
        {
            'temperature': heatmap_data['temperature'],
            'rainfall': heatmap_data['rainfall']
        }, 
        geometry=points,
        crs="EPSG:4326"
    )
    
    # Spatial join to find which state each point belongs to
    joined = gpd.sjoin(points_gdf, india_states, how="left", predicate="within")
    
    # Calculate regional averages
    state_name_field = [col for col in india_states.columns if 'name' in col.lower() or 'state' in col.lower()]
    if state_name_field:
        state_field = state_name_field[0]
    else:
        state_field = india_states.columns[0]  # Fallback
    
    # Group by state and calculate averages
    regional_stats = joined.groupby(state_field).agg({
        'temperature': ['mean', 'min', 'max', 'std'],
        'rainfall': ['mean', 'min', 'max', 'std']
    }).reset_index()
    
    # Flatten the multi-index columns
    regional_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in regional_stats.columns]
    
    return regional_stats

# Plot heatmap with state boundaries
def plot_heatmap(data, metric, title, india_states=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    values = data[metric]
    
    if metric == 'rainfall':
        cmap = LinearSegmentedColormap.from_list('rainfall_cmap', ['#f7fbff', '#08306b'])
        vmin, vmax = np.min(values) * 0.9, np.max(values) * 1.1
        cb_label = 'Rainfall (mm)'
    else:
        cmap = LinearSegmentedColormap.from_list('temp_cmap', ['#ffffcc', '#730707'])
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
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig


# NEW FUNCTION: Calculate climate metrics
def calculate_climate_metrics(climate_df, current_year, heatmap_data):
    # Initialize metrics dictionary
    metrics = {}
    
    # Current year average temperature and rainfall
    metrics["current_temp_avg"] = np.mean(heatmap_data['temperature'])
    metrics["current_rain_avg"] = np.mean(heatmap_data['rainfall'])
    
    # Find hottest and wettest locations
    hottest_idx = np.argmax(heatmap_data['temperature'])
    wettest_idx = np.argmax(heatmap_data['rainfall'])
    
    metrics["hottest_location"] = {
        "temp": heatmap_data['temperature'][hottest_idx],
        "lat": heatmap_data['lats'][hottest_idx],
        "lon": heatmap_data['lons'][hottest_idx]
    }
    
    metrics["wettest_location"] = {
        "rain": heatmap_data['rainfall'][wettest_idx],
        "lat": heatmap_data['lats'][wettest_idx],
        "lon": heatmap_data['lons'][wettest_idx]
    }
    
    # Historical comparison (10-year average)
    historical_years = sorted(climate_df['year'].unique())
    baseline_year = max(min(historical_years), current_year - 10)
    
    historical_data = climate_df[(climate_df['year'] >= baseline_year) & 
                                (climate_df['year'] < baseline_year + 10)]
    
    if not historical_data.empty:
        metrics["historical_temp_avg"] = historical_data['temperature'].mean()
        metrics["historical_rain_avg"] = historical_data['rainfall'].mean()
        
        # Calculate anomalies
        metrics["temp_anomaly"] = metrics["current_temp_avg"] - metrics["historical_temp_avg"]
        metrics["rain_anomaly_pct"] = ((metrics["current_rain_avg"] / metrics["historical_rain_avg"]) - 1) * 100
    else:
        # Fallbacks if historical data not available
        metrics["historical_temp_avg"] = None
        metrics["historical_rain_avg"] = None
        metrics["temp_anomaly"] = None
        metrics["rain_anomaly_pct"] = None
    
    # Extreme indices
    if not historical_data.empty:
        temp_std = historical_data['temperature'].std()
        metrics["extreme_heat_pct"] = np.mean(heatmap_data['temperature'] > 
                                            metrics["historical_temp_avg"] + 2*temp_std) * 100
    else:
        metrics["extreme_heat_pct"] = None
    
    # Year-over-year changes
    prev_year = current_year - 1
    prev_year_data = climate_df[climate_df['year'] == prev_year]
    
    if not prev_year_data.empty:
        metrics["yoy_temp_change"] = metrics["current_temp_avg"] - prev_year_data['temperature'].mean()
        metrics["yoy_rain_change_pct"] = ((metrics["current_rain_avg"] / prev_year_data['rainfall'].mean()) - 1) * 100
    else:
        metrics["yoy_temp_change"] = None
        metrics["yoy_rain_change_pct"] = None
    
    return metrics

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
    india_states = load_india_states()
    
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
        
        show_states = st.sidebar.checkbox("Show State Boundaries", value=True)
        
        # Load embedding and process data
        embedding = load_embedding(selected_year)
        heatmap_data = process_climate_data(embedding, coordinates_df, climate_df, selected_year)
        
        # Display year info
        st.header(f"Climate Data for {selected_year}")
        if selected_year > max(historical_years):
            st.info("⚠️ Data shown is predicted based on graph embeddings or projection models.")
        else:
            st.info("📊 Historical data from records.")
        
        # Calculate climate metrics
        climate_metrics = calculate_climate_metrics(climate_df, selected_year, heatmap_data)
        
        # Dashboard Metrics Section
        st.subheader("📊 Climate Dashboard")
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric(
                "Average Temperature", 
                f"{climate_metrics['current_temp_avg']:.2f}°C",
                f"{climate_metrics.get('temp_anomaly', 0):.2f}°C vs 10yr avg" 
                if climate_metrics.get('temp_anomaly') is not None else None,
                delta_color="inverse" 
            )
        
        with metric_cols[1]:
            st.metric(
                "Average Rainfall", 
                f"{climate_metrics['current_rain_avg']:.2f} mm",
                f"{climate_metrics.get('rain_anomaly_pct', 0):.1f}% vs 10yr avg" 
                if climate_metrics.get('rain_anomaly_pct') is not None else None
            )
        
        
        
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Temperature Map")
            states_for_plot = india_states if show_states else None
            temp_fig = plot_heatmap(heatmap_data, 'temperature', f"Temperature Distribution ({selected_year})", states_for_plot)
            st.pyplot(temp_fig)
        
        with col2:
            st.subheader("Rainfall Map")
            states_for_plot = india_states if show_states else None
            rain_fig = plot_heatmap(heatmap_data, 'rainfall', f"Rainfall Distribution ({selected_year})", states_for_plot)
            st.pyplot(rain_fig)
        
        # Year-over-Year Changes
        yoy_cols = st.columns(2)
        with yoy_cols[0]:
            if climate_metrics.get('yoy_temp_change') is not None:
                st.metric(
                    "Temp Change from Previous Year", 
                    f"{climate_metrics['yoy_temp_change']:.2f}°C",
                    delta=None,
                )
        
        with yoy_cols[1]:
            if climate_metrics.get('yoy_rain_change_pct') is not None:
                st.metric(
                    "Rainfall Change from Previous Year", 
                    f"{climate_metrics['yoy_rain_change_pct']:.1f}%",
                    delta=None,
                )
        
        # Regional Statistics
        st.subheader("📍 Regional Statistics")
        
        # Calculate regional averages if state boundaries are available
        if india_states is not None:
            regional_stats = calculate_regional_averages(heatmap_data, india_states)
            
            # Display top 5 hottest and wettest states
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Hottest States")
                hottest_states = regional_stats.sort_values('temperature_mean', ascending=False).head(10)
                
                for i, row in hottest_states.iterrows():
                    state_name = row[regional_stats.columns[0]]  # First column contains state names
                    temp = row['temperature_mean']
                    st.markdown(f"**{state_name}**: {temp:.2f}°C")
            
            with col2:
                st.subheader("Top 10 Wettest States")
                wettest_states = regional_stats.sort_values('rainfall_mean', ascending=False).head(10)
                
                for i, row in wettest_states.iterrows():
                    state_name = row[regional_stats.columns[0]]  # First column contains state names
                    rain = row['rainfall_mean']
                    st.markdown(f"**{state_name}**: {rain:.2f} mm")
            
            # Full regional data in expandable section
            with st.expander("View All Regional Statistics"):
                # Format the dataframe for display
                display_cols = [regional_stats.columns[0], 'temperature_mean', 'temperature_min', 
                               'temperature_max', 'rainfall_mean', 'rainfall_min', 'rainfall_max']
                display_df = regional_stats[display_cols].copy()
                
                # Rename columns for better readability
                display_df.columns = ['State', 'Avg Temp (°C)', 'Min Temp (°C)', 
                                     'Max Temp (°C)', 'Avg Rain (mm)', 'Min Rain (mm)', 'Max Rain (mm)']
                
                # Round numeric columns
                for col in display_df.columns[1:]:
                    display_df[col] = display_df[col].round(2)
                
                st.dataframe(display_df)
        
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