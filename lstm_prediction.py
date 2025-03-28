import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class AdvancedPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.short_term_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                                        batch_first=True, dropout=0.3)
        self.long_term_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                                       batch_first=True, dropout=0.3)
        
        self.trend_magnitude = nn.Parameter(torch.tensor(0.05))
        self.variation_scale = nn.Parameter(torch.tensor(0.01))
        
        # Corrected context embedding dimension
        self.context_embedding = nn.Sequential(
            nn.Linear(10, hidden_dim * 2),  # Output matches combined_out dimension
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, context_features=None):
        x_projected = self.input_projection(x)
        
        short_out, _ = self.short_term_lstm(x_projected)
        long_out, _ = self.long_term_lstm(x_projected)
        
        short_out = short_out[:, -1, :]
        long_out = long_out[:, -1, :]
        
        combined_out = torch.cat([short_out, long_out], dim=1)
        
        if context_features is not None:
            # Remove singleton dimensions from context features
            if context_features.dim() == 3:
                context_features = context_features.squeeze(1)
            
            context_embed = self.context_embedding(context_features)
            combined_out += context_embed  # Now [batch_size, 256] + [batch_size, 256]
        
        output = self.fc_layers(combined_out)
        
        trend = torch.randn_like(output) * self.trend_magnitude
        variation = torch.randn_like(output) * self.variation_scale
        
        return output + trend + variation

def predict_future(embeddings, target_year, context_data=None):
    years = list(embeddings.keys())
    current_year = max(years)

    # Prepare training data
    train_x = [torch.stack([embeddings[y] for y in range(year, year + 5)]) for year in years[:-5]]
    train_y = [embeddings[year + 5] for year in years[:-5]]

    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)

    num_samples, window_size, num_nodes, features = train_x.shape
    train_x = train_x.reshape(num_samples, window_size, -1)  # Flatten nodes and features

    input_dim = train_x.shape[-1]
    
    # Advanced model
    model = AdvancedPredictionModel(input_dim)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Sophisticated loss function
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    def combined_loss(pred, target):
        # Reshape target to match prediction
        target = target.reshape(pred.shape)
        return mse_loss(pred, target) + 0.2 * l1_loss(pred, target)

    # Prepare context data
    if context_data is None:
        context_data = [torch.randn(1, 10) for _ in range(num_samples)]
    else:
        # Ensure context data matches batch size
        context_data = context_data[:num_samples]
    
    # Extended training
    epochs = 600
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Prepare context for this batch
        context = torch.stack(context_data)
        
        predictions = model(train_x, context)
        predictions = predictions.reshape(num_samples, num_nodes, features)
        
        loss = combined_loss(predictions, train_y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "advanced_prediction_model.pth")

    print("Advanced training completed")

    # Prediction phase

    model.load_state_dict(torch.load("advanced_prediction_model.pth"))
    model.eval()

    future_embeddings = embeddings.copy()
    for year in range(current_year + 1, target_year + 1):
        past_5_years = [future_embeddings[y] for y in range(year - 5, year)]
        input_seq = torch.stack(past_5_years).unsqueeze(0)
        input_seq = input_seq.reshape(1, 5, -1)  # Flatten nodes and features
        
        # Use a context feature for prediction
        context = torch.randn(1, 10)

        with torch.no_grad():
            predicted = model(input_seq, context).squeeze(0).reshape(num_nodes, features)
        
        future_embeddings[year] = predicted
        torch.save(predicted, os.path.join("graph_embeddings", f"embeddings_{year}.pt"))
        print(f"Predictions for {year} saved")

    return future_embeddings

if __name__ == "__main__":
    # Load embeddings
    embeddings = {}
    for year in range(2000, 2024):
        embeddings[year] = torch.load(os.path.join("graph_embeddings", f"embeddings_{year}.pt"))
    
    target_year = int(input("Enter the target year for predictions: "))
    predict_future(embeddings, target_year)