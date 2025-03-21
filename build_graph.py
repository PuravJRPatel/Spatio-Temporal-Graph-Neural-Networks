import pandas as pd
import numpy as np
from torch_geometric.data import Data
from scipy.spatial import cKDTree
import numpy as np
import torch
import os

def load_data(filepath):
  return pd.read_csv(filepath)

def build_graph(df_year, k=8):
  node_features = torch.tensor(df_year[['rainfall','temperature']].values, dtype=torch.float)
  node_positions = df_year[['latitude','longitude']].values

  kdtree = cKDTree(node_positions)
  edges = kdtree.query(node_positions, k = k+1)[1]

  edge_index = []
  edge_features = []

  for i, neighbors in enumerate(edges):
      for neighbor in neighbors:
          if i != neighbor:
              edge_index.append((i, neighbor))

              dist = np.linalg.norm(node_positions[i] - node_positions[neighbor])  
              rainfall_diff = abs(node_features[i][0] - node_features[neighbor][0])  
              temperature_diff = abs(node_features[i][1] - node_features[neighbor][1]) 

              edge_features.append([dist, rainfall_diff, temperature_diff])

  edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
  edge_attr = torch.tensor(edge_features, dtype=torch.float)

  return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

def save_graph(graph, year, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(graph, os.path.join(output_dir, f"graph_{year}.pt"))

def main():
    data_path = r"C:\Users\purav\OneDrive\Desktop\STGNN\Spatio-Temporal-Graph-Neural-Networks\Processed Data\climate_data.csv"
    output_dir = "processed_graphs"

    df = load_data(data_path)
    years = df['year'].unique()

    for year in years:
        print(f"Processing year {year}...")
        df_year = df[df['year'] == year]
        graph = build_graph(df_year)
        save_graph(graph, year, output_dir)

    print("Graph processing complete")

if __name__ == "__main__":
    main()
    
