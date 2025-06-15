import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
import os

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def load_graph(year, graph_dir):
    graph_path = os.path.join(graph_dir, f"graph_{year}.pt")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file for year {year} not found in {graph_dir}")
    
    return torch.load(graph_path, weights_only= False)

def train_model(graph, hidden_dim, output_dim, num_epochs, learning_rate):
    model = GAT(in_channels=graph.x.shape[1], hidden_channels=hidden_dim, out_channels=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    # Split data: 80% for training, 20% for validation
    num_nodes = graph.x.shape[0]
    train_size = int(0.8 * num_nodes)
    perm = torch.randperm(num_nodes)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    train_x, val_x = graph.x[train_idx], graph.x[val_idx]

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(graph.x, graph.edge_index)
        
        train_loss = F.mse_loss(output[train_idx], train_x)
        train_loss.backward()
        optimizer.step()

        # Validation loss (without gradient updates)
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(output[val_idx], val_x)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

    return model, output, train_losses, val_losses

def save_embeddings(output, year, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)
    torch.save(output, os.path.join(embeddings_dir, f"embeddings_{year}.pt"))
    print(f"Embeddings saved for {year}")

def plot_average_losses(all_train_losses, all_val_losses, num_epochs, loss_plot_dir):
    avg_train_loss = [sum(losses) / len(losses) for losses in zip(*all_train_losses)]
    avg_val_loss = [sum(losses) / len(losses) for losses in zip(*all_val_losses)]

    epochs = range(1, num_epochs + 1)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_train_loss, label='Avg Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Average Training Loss (1980–2023)')
    plt.grid(True)
    plt.tight_layout()
    train_loss_path = os.path.join(loss_plot_dir, "avg_train_loss.png")
    plt.savefig(train_loss_path)
    plt.close()
    print(f"Saved average train loss plot at {train_loss_path}")

    # Plot validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_val_loss, label='Avg Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Average Validation Loss (1980–2023)')
    plt.grid(True)
    plt.tight_layout()
    val_loss_path = os.path.join(loss_plot_dir, "avg_val_loss.png")
    plt.savefig(val_loss_path)
    plt.close()
    print(f"Saved average val loss plot at {val_loss_path}")


def main():
    graph_dir = "backend/data/processed_graphs"
    embeddings_dir = "backend/data/graph_embeddings"
    loss_plot_dir = "backend/data/loss_plots"
    os.makedirs(loss_plot_dir, exist_ok=True)

    hidden_dim = 128
    output_dim = 2
    num_epochs = 1000
    learning_rate = 0.001

    all_train_losses = []
    all_val_losses = []

    for year in range(1980, 2024):
        print(f"\nTraining GAT for year {year}")
        graph = load_graph(year, graph_dir)
        _, output, train_losses, val_losses = train_model(graph, hidden_dim, output_dim, num_epochs, learning_rate)
        save_embeddings(output, year, embeddings_dir)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

    plot_average_losses(all_train_losses, all_val_losses, num_epochs, loss_plot_dir)
    print("GAT Training Completed")

if __name__ == "__main__":
    main()
