import torch
import torch.nn as nn
import torch.optim as optim
import os

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)  

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output
        

graph_dir = "Spatio-Temporal-Graph-Neural-Networks\graph_embeddings"

def load_embeddings():
    embeddings = {}
    for year in range(2000, 2024):
        embeddings[year] = torch.load(os.path.join(graph_dir, f"embeddings_{year}.pt"))
    return embeddings

def predict_future(embeddings, target_year):
    years = list(embeddings.keys())
    current_year = max(years)

    train_x = [torch.stack([embeddings[y] for y in range(year, year + 10)]) for year in years[:-10]]
    train_y = [embeddings[year + 10] for year in years[:-10]]

    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)

    num_samples, window_size, num_nodes, features = train_x.shape

    train_x = train_x.reshape(num_samples, window_size, num_nodes * features)

    input_dim = train_x.shape[-1]  
    model = LSTMModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    epochs = 500
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(train_x)
        predictions = predictions.reshape(num_samples, num_nodes, features)
        loss = criterion(predictions, train_y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), "lstm_model.pth")
    print("LSTM training completed and model saved")

    model.eval()
    future_embeddings = embeddings.copy()
    for year in range(current_year + 1, target_year + 1):
        past_10_years = [future_embeddings[y] for y in range(year - 10, year)]
        input_seq = torch.stack(past_10_years).unsqueeze(0).reshape(1, window_size, num_nodes * features)

        predicted = model(input_seq).squeeze(0).reshape(num_nodes, features)
        future_embeddings[year] = predicted
        torch.save(predicted, f"{graph_dir}/embeddings_{year}.pt")
        print(f"Predictions for {year} saved")

if __name__ == "__main__":
    embeddings = load_embeddings()
    target_year = int(input("Enter the target year for predictions: "))
    predict_future(embeddings, target_year)
