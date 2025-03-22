import os
import torch
from build_graph import load_data, build_graph, save_graph
from train_gcn import load_graph, train_model, save_embeddings
from lstm_prediction import load_embeddings, predict_future

data_path = "Processed Data/climate_data.csv"
graph_dir = "processed_graphs"
embeddings_dir = "graph_embeddings"
target_year = 2024 


if not os.path.exists(graph_dir) or len(os.listdir(graph_dir)) == 0:
    print("Building graphs...")
    df = load_data(data_path)
    years = df['year'].unique()
    for year in years:
        df_year = df[df['year'] == year]
        graph = build_graph(df_year)
        save_graph(graph, year, graph_dir)
    print("Graph processing complete.")
else:
    print("Graphs already exist. Skipping graph building.")


if not os.path.exists(embeddings_dir) or len(os.listdir(embeddings_dir)) == 0:
    print("Training GCN and generating embeddings...")
    hidden_dim = 20
    output_dim = 2
    num_epochs = 400
    learning_rate = 0.01

    for year in range(2000, 2024):
        print(f"Training GCN for year {year}")
        graph = load_graph(year, graph_dir)
        _, output = train_model(graph, hidden_dim, output_dim, num_epochs, learning_rate)
        save_embeddings(output, year, embeddings_dir)
    print("GCN training complete.")
else:
    print("Embeddings already exist. Skipping GCN training.")

embeddings = load_embeddings()
predict_future(embeddings, target_year)

print("Pipeline execution complete.")
