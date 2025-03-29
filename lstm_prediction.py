import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

class AdvancedPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.short_term_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                                        batch_first=True, dropout=0.3)
        self.long_term_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                                       batch_first=True, dropout=0.3)
        
        self.trend_magnitude = nn.Parameter(torch.tensor(0.3))
        self.variation_scale = nn.Parameter(torch.tensor(0.05))
        self.upward_bias = nn.Parameter(torch.tensor(0.1))
        self.smoothing_factor = nn.Parameter(torch.tensor(0.9))
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

        self.register_buffer('prev_prediction', None)

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
            combined_out += context_embed  
        
        output = self.fc_layers(combined_out)        
        trend = (torch.randn_like(output) * 0.5 + 0.5) * self.trend_magnitude
        variation = torch.randn_like(output) * self.variation_scale
        upward = torch.abs(output) * self.upward_bias
        raw_prediction = output + trend + variation + upward
        if self.prev_prediction is not None and self.training == False:
            smoothed_prediction = self.smoothing_factor * raw_prediction + (1 - self.smoothing_factor) * self.prev_prediction
            self.prev_prediction = smoothed_prediction.detach()  # Update for next prediction
            return smoothed_prediction
        else:
            # First prediction or during training
            if self.training == False:
                self.prev_prediction = raw_prediction.detach()  # Store for next prediction
            return raw_prediction

def ensure_directory_exists(directory):
    """Ensure the directory exists, create if necessary."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    return True

def safe_save_model(model, file_path):
    """Safely save model with error handling."""
    try:
        # Get the directory part
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the model
        torch.save(model.state_dict(), file_path)
        print(f"Model saved to: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving model to {file_path}: {e}")
        # Try saving to current directory as fallback
        try:
            fallback_path = os.path.basename(file_path)
            torch.save(model.state_dict(), fallback_path)
            print(f"Model saved to fallback location: {fallback_path}")
            return fallback_path
        except Exception as e2:
            print(f"Failed to save model to fallback location: {e2}")
            return False

def predict_future(embeddings, target_year, context_data=None):
    years = list(embeddings.keys())
    current_year = max(years)

    # Create output directory if it doesn't exist
    model_dir = os.path.join(os.getcwd(), "models")
    ensure_directory_exists(model_dir)
    model_path = os.path.join(model_dir, "advanced_prediction_model.pth")
    
    # Also ensure the graph_embeddings directory exists
    embeddings_dir = os.path.join(os.getcwd(), "graph_embeddings")
    if not ensure_directory_exists(embeddings_dir):
        print("Warning: Could not create/access embeddings directory. Using current directory instead.")
        embeddings_dir = os.getcwd()

    # Prepare training data
    train_x = [torch.stack([embeddings[y] for y in range(year, year + 10)]) for year in years[:-10]]
    train_y = [embeddings[year + 10] for year in years[:-10]]

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
        base_loss =  0.8 * mse_loss(pred, target) + 0.2 * l1_loss(pred, target)
        batch_size = pred.shape[0]
        if batch_size > 1:
        # Calculate differences between consecutive items in batch
            diffs = pred[1:] - pred[:-1]
        
        # Penalize negative trends (encourage positive trends)
            neg_trend_penalty = torch.mean(torch.relu(-diffs))
        
        # Add to base loss
            return base_loss + 0.2 * neg_trend_penalty
        else:
            return base_loss
    # Prepare context data
    if context_data is None:
        context_data = [torch.randn(1, 10) for _ in range(num_samples)]
    else:
        # Ensure context data matches batch size
        context_data = context_data[:num_samples]
    
    # Extended training
    epochs = 1000
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
            # Save model with error handling
            saved = safe_save_model(model, model_path)
            if not saved:
                print("Warning: Could not save best model. Continuing training...")

    print("Advanced training completed")

    # Check if model file exists before loading
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded best model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using latest trained model instead.")
    else:
        print(f"No saved model found at {model_path}. Using latest trained model.")

    model.eval()

    future_embeddings = embeddings.copy()
    last_prediction = None
    min_allowed_change = -0.01

    for year in range(current_year + 1, target_year + 1):
        past_10_years = [future_embeddings[y] for y in range(year - 10, year)]
        input_seq = torch.stack(past_10_years).unsqueeze(0)
        
        # Use the same reshaping logic as in training
        _, seq_len, nodes, feat = input_seq.shape
        input_seq = input_seq.reshape(1, seq_len, -1)  # Reshape consistently with training
        
        # Use a context feature for prediction
        context = torch.randn(1, 10)

        with torch.no_grad():
            predicted = model(input_seq, context)
            # Reshape the output back to the original dimensions
            predicted = predicted.reshape(num_nodes, features)
        if last_prediction is not None:
            # Check each feature and limit drops
            change = predicted - last_prediction
            excessive_drops = (change < min_allowed_change)
            
            # Where drops are excessive, limit them
            if excessive_drops.any():
                predicted[excessive_drops] = last_prediction[excessive_drops] + min_allowed_change
        
        last_prediction = predicted.clone()
        future_embeddings[year] = predicted
        
        # Save the prediction with error handling
        embedding_file = os.path.join(embeddings_dir, f"embeddings_{year}.pt")
        try:
            torch.save(predicted, embedding_file)
            print(f"Predictions for {year} saved to {embedding_file}")
        except Exception as e:
            print(f"Error saving prediction for year {year}: {e}")
            # Try alternative location
            try:
                alt_file = f"embeddings_{year}.pt"
                torch.save(predicted, alt_file)
                print(f"Predictions for {year} saved to alternative location: {alt_file}")
            except Exception as e2:
                print(f"Failed to save predictions for year {year}: {e2}")

    return future_embeddings

if __name__ == "__main__":
    try:
        # Load embeddings with error handling
        embeddings = {}
        embeddings_dir = os.path.join(os.getcwd(), "graph_embeddings")
        
        if not os.path.exists(embeddings_dir):
            print(f"Warning: Directory {embeddings_dir} does not exist.")
            print("Please ensure the graph_embeddings directory exists and contains embedding files.")
            sys.exit(1)
            
        missing_years = []
        for year in range(1980, 2024):
            embedding_file = os.path.join(embeddings_dir, f"embeddings_{year}.pt")
            if os.path.exists(embedding_file):
                try:
                    embeddings[year] = torch.load(embedding_file)
                except Exception as e:
                    print(f"Error loading embeddings for year {year}: {e}")
                    missing_years.append(year)
            else:
                missing_years.append(year)
        
        if missing_years:
            print(f"Warning: Embeddings missing for years: {missing_years}")
            if len(missing_years) > 10:
                print("Too many missing embeddings. Please check your data.")
                sys.exit(1)
        
        target_year = int(input("Enter the target year for predictions: "))
        predict_future(embeddings, target_year)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
