import streamlit as st
import plotly.express as px
import pandas as pd
import folium
from streamlit_folium import st_folium

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("backend/data/Processed Data/climate_data.csv")

data = load_data()

# Sidebar Options
years = sorted(data['year'].unique())
st.sidebar.title("Visualization Options")
selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1)

data_type = st.sidebar.radio("Select Data Type", ["Temperature", "Rainfall"])

# Line Graph: Actual vs Predicted
def plot_line_graph():
    df_filtered = data[['year', 'temperature', 'predicted_temperature', 'rainfall', 'predicted_rainfall']]
    
    if data_type == "Temperature":
        fig = px.line(df_filtered, x='year', y=['temperature', 'predicted_temperature'],
                      labels={'value': 'Temperature (°C)', 'variable': 'Data Type'},
                      title=f"Actual vs Predicted {data_type}")
    else:
        fig = px.line(df_filtered, x='year', y=['rainfall', 'predicted_rainfall'],
                      labels={'value': 'Rainfall (mm)', 'variable': 'Data Type'},
                      title=f"Actual vs Predicted {data_type}")
    
    st.plotly_chart(fig)

# Map Visualization
def plot_map():
    st.subheader(f"{data_type} Distribution - {selected_year}")
    df_year = data[data['year'] == selected_year]
    
    # Base Map
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    
    for _, row in df_year.iterrows():
        value = row['temperature'] if data_type == "Temperature" else row['rainfall']
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color="red" if data_type == "Temperature" else "blue",
            fill=True,
            fill_opacity=0.6,
            popup=f"{data_type}: {value}"
        ).add_to(m)
    
    st_folium(m, width=700, height=500)

# Layout
st.title("Climate Data Visualization")
plot_line_graph()
st.markdown("---")
plot_map()
