import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
def load_graph(year, graph_dir):
    graph_path = os.path.join(graph_dir, f"graph_{year}.pt")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file for year {year} not found in {graph_dir}")
    
    return torch.load(graph_path)

def train_model(graph, hidden_dim, output_dim, num_epochs, learning_rate):
    model = GCN(in_channels=graph.x.shape[1], hidden_channels=hidden_dim, out_channels=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(graph.x, graph.edge_index)
        loss = F.mse_loss(output,graph.x)
        loss.backward()
        optimizer.step()

        if epoch%100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return model,output

def save_embeddings(output, year, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)
    torch.save(output, os.path.join(embeddings_dir, f"embeddings_{year}.pt"))
    print(f"Embeddings saved for {year}")

def main():
    graph_dir = "processed_graphs"
    embeddings_dir = "graph_embeddings"
    
    hidden_dim = 32
    output_dim = 2

    num_epochs = 1000
    learning_rate = 0.01

    for year in range(2000, 2024):
        print(f"\nTraining GCN for year {year}")
        graph = load_graph(year, graph_dir)
        _, output = train_model(graph, hidden_dim, output_dim, num_epochs, learning_rate)
        save_embeddings(output, year, embeddings_dir)

    print("GCN Training Completed")


if __name__ == "__main__":
    main()

    