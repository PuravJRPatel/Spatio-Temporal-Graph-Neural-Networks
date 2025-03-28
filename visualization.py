import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import time
import pydeck as pdk
from build_graph import build_graph, save_graph, load_data
from train_gcn import load_graph, train_model, save_embeddings
from lstm_prediction import load_embeddings, predict_future

GRAPH_DIR = "processed_graphs"
EMBEDDINGS_DIR = "graph_embeddings"
DATA_PATH = "Processed Data/climate_data.csv"

def ensure_graph(year):
    graph_path = f"{GRAPH_DIR}/graph_{year}.pt"
    if not os.path.exists(graph_path):
        df = load_data(DATA_PATH)
        df_year = df[df["year"] == (2023 if year > 2023 else year)]
        graph = build_graph(df_year)
        save_graph(graph, year, GRAPH_DIR)
    return torch.load(graph_path, weights_only=True)

def ensure_embeddings(year):
    embed_path = f"{EMBEDDINGS_DIR}/embeddings_{year}.pt"
    if not os.path.exists(embed_path):
        if year > 2023:
            embeddings = load_embeddings()
            predict_future(embeddings, year)
        else:
            graph = ensure_graph(year)
            _, output = train_model(graph, hidden_dim=32, output_dim=2, num_epochs=1000, learning_rate=0.01)
            save_embeddings(output, year, EMBEDDINGS_DIR)
    return torch.load(embed_path, weights_only=True).detach().numpy()

st.title("India Climate Predictions using GCN + LSTM")
selected_year = st.number_input("Enter the prediction year (>=2024):", min_value=2024, value=2024, step=1)

all_embeddings = {}
for year in range(2000, selected_year + 1):
    all_embeddings[year] = ensure_embeddings(year)

df = pd.read_csv(DATA_PATH)
df = df[df['year'] == 2023].reset_index(drop=True)

latitudes = df["latitude"].values
longitudes = df["longitude"].values

animation_type = st.radio("Select Heatmap Type:", ("Rainfall", "Temperature"))

def prepare_pydeck_data(year):
    embedding = all_embeddings[year]
    values = embedding[:, 0] if animation_type == "Rainfall" else embedding[:, 1]
    
    return pd.DataFrame({
        "latitude": latitudes,
        "longitude": longitudes,
        "value": values,
    })

def create_pydeck_map(year):
    data = prepare_pydeck_data(year)

    layer = pdk.Layer(
        "HeatmapLayer",
        data,
        get_position=["longitude", "latitude"],
        get_weight="value",
        radius_pixels=30,
        intensity=1,
        color_range=[
            [0, 0, 255, 100],  # Blue (Cool)
            [0, 255, 255, 150], # Cyan
            [0, 255, 0, 200],   # Green
            [255, 255, 0, 250], # Yellow
            [255, 0, 0, 255]    # Red (Hot)
        ],
    )

    view_state = pdk.ViewState(
        latitude=np.mean(latitudes),
        longitude=np.mean(longitudes),
        zoom=4.5,  # Keep India Enlarged
        pitch=40,
    )

    return pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="light")

st.subheader("Heatmap Transition")
play_button = st.button("Play")
pause_button = st.button("Pause")

map_container = st.empty()
current_year_text = st.empty()

playing = False
if play_button:
    playing = True
if pause_button:
    playing = False

for year in range(2000, selected_year + 1):
    if not playing:
        break

    current_year_text.subheader(f"Year: {year}")
    map_container.pydeck_chart(create_pydeck_map(year))
    time.sleep(0.5)