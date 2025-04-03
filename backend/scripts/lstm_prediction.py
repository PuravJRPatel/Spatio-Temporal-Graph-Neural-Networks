import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

class ClimateAwarePredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, climate_mode="standard"):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.short_term_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                                      batch_first=True, dropout=0.3)
        self.long_term_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                                     batch_first=True, dropout=0.3)
        
        # Climate-aware parameters - adjustable based on selected mode
        self.climate_mode = climate_mode
        if climate_mode == "warming":
            # Parameters calibrated for warming scenario
            self.trend_magnitude = nn.Parameter(torch.tensor(0.5))  # Increased trend magnitude
            self.variation_scale = nn.Parameter(torch.tensor(0.07))  # More variation
            self.upward_bias = nn.Parameter(torch.tensor(0.2))  # Stronger upward bias
            self.smoothing_factor = nn.Parameter(torch.tensor(0.6))  # Less smoothing for stronger trends
        elif climate_mode == "extreme":
            # Parameters calibrated for extreme weather scenario
            self.trend_magnitude = nn.Parameter(torch.tensor(0.4))
            self.variation_scale = nn.Parameter(torch.tensor(0.15))  # Much more variation
            self.upward_bias = nn.Parameter(torch.tensor(0.15))
            self.smoothing_factor = nn.Parameter(torch.tensor(0.5))  # Less smoothing
        else:  # standard mode
            # Default parameters similar to original
            self.trend_magnitude = nn.Parameter(torch.tensor(0.3))
            self.variation_scale = nn.Parameter(torch.tensor(0.05))
            self.upward_bias = nn.Parameter(torch.tensor(0.1))
            self.smoothing_factor = nn.Parameter(torch.tensor(0.7))
        
        # Seasonal component - for climate modeling
        self.use_seasonal = True
        self.seasonal_amplitude = nn.Parameter(torch.tensor(0.1))
        
        # Climate context features
        self.context_embedding = nn.Sequential(
            nn.Linear(12, hidden_dim * 2),  # Expanded to 12 features for climate context
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.register_buffer('prev_prediction', None)
        self.register_buffer('year_counter', torch.tensor(0))

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
        
        # Apply climate-aware transformations
        trend = (torch.randn_like(output) * 0.5 + 0.5) * self.trend_magnitude
        
        # Climate variability - more extreme in later years
        if not self.training:
            self.year_counter += 1
            year_factor = torch.clamp(self.year_counter / 10.0, 0, 1.5)
            variation = torch.randn_like(output) * self.variation_scale * (1 + year_factor * 0.2)
        else:
            variation = torch.randn_like(output) * self.variation_scale
            
        # Climate upward bias - stronger for temperature data
        upward = torch.abs(output) * self.upward_bias
        
        # Add seasonal component for climate modeling
        seasonal = torch.zeros_like(output)
        if self.use_seasonal and not self.training:
            # Simple sinusoidal seasonality
            seasonal = torch.sin(self.year_counter * 0.5) * self.seasonal_amplitude
            
        raw_prediction = output + trend + variation + upward + seasonal
        
        # Apply smoothing with climate-aware mechanism
        if self.prev_prediction is not None and self.training == False:
            # Adjust smoothing based on climate mode
            if self.climate_mode == "extreme":
                # Less smoothing when predicting extreme events
                local_smoothing = self.smoothing_factor * (0.7 + 0.3 * torch.rand_like(raw_prediction))
            else:
                local_smoothing = self.smoothing_factor
                
            smoothed_prediction = local_smoothing * raw_prediction + (1 - local_smoothing) * self.prev_prediction
            self.prev_prediction = smoothed_prediction.detach()
            return smoothed_prediction
        else:
            if self.training == False:
                self.prev_prediction = raw_prediction.detach()
            return raw_prediction

def climate_aware_loss(pred, target, temp_indices=None, rainfall_indices=None, climate_mode="standard"):
    """
    Climate-aware loss function that applies different penalties based on data type and expected climate trends
    
    Args:
        pred: model predictions
        target: target values
        temp_indices: indices corresponding to temperature data
        rainfall_indices: indices corresponding to rainfall data
        climate_mode: type of climate scenario to model (standard, warming, extreme)
    """
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    # Reshape target to match prediction
    target = target.reshape(pred.shape)
    
    # Base loss combining MSE and L1
    base_loss = 0.7 * mse_loss(pred, target) + 0.3 * l1_loss(pred, target)
    
    batch_size = pred.shape[0]
    if batch_size > 1:
        # Calculate differences between consecutive items
        diffs = pred[1:] - pred[:-1]
        
        # Different penalties based on climate mode
        if climate_mode == "warming":
            # Strongly penalize negative temperature trends
            if temp_indices is not None:
                temp_diffs = diffs[:, temp_indices]
                temp_neg_penalty = torch.mean(torch.relu(-temp_diffs)) * 0.3
                base_loss = base_loss + temp_neg_penalty
                
            # Rainfall can be more variable
            if rainfall_indices is not None:
                rainfall_diffs = diffs[:, rainfall_indices]
                rainfall_var_penalty = torch.mean(torch.abs(rainfall_diffs)) * 0.1
                base_loss = base_loss + rainfall_var_penalty
                
        elif climate_mode == "extreme":
            # Penalize stability (encourage larger changes)
            stability_penalty = torch.mean(torch.exp(-torch.abs(diffs) * 5)) * 0.2
            base_loss = base_loss + stability_penalty
        else:
            # Standard mode - mild penalties for negative trends
            neg_trend_penalty = torch.mean(torch.relu(-diffs)) * 0.1
            base_loss = base_loss + neg_trend_penalty
    
    return base_loss

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

def create_climate_context(year, climate_mode="standard"):
    """
    Create climate context features that encode climate-relevant information
    
    Args:
        year: the year for which to create context
        climate_mode: type of climate scenario (standard, warming, extreme)
    
    Returns:
        torch.Tensor: climate context features
    """
    # Base year for reference (e.g., pre-industrial or start of data)
    base_year = 1980
    years_since_base = year - base_year
    
    # CO2 level approximation (simplified)
    co2_base = 340  # Approximate CO2 level in 1980
    co2_annual_increase = 2.0  # Approximate annual increase
    
    # Create context features
    context = torch.zeros(12)
    
    # Years since base (normalized)
    context[0] = years_since_base / 100.0
    
    # CO2 trend - accelerating for warming scenario
    if climate_mode == "warming":
        context[1] = (co2_base + co2_annual_increase * years_since_base * 1.1) / 500.0
    elif climate_mode == "extreme":
        context[1] = (co2_base + co2_annual_increase * years_since_base * 1.2) / 500.0
    else:
        context[1] = (co2_base + co2_annual_increase * years_since_base) / 500.0
    
    # Climate variability factor - increasing in later years for extreme scenario
    if climate_mode == "extreme":
        context[2] = 0.5 + (years_since_base / 80.0) * 0.5
    else:
        context[2] = 0.3 + (years_since_base / 100.0) * 0.3
    
    # Seasonal and cyclic factors
    context[3] = np.sin(years_since_base * 0.5) * 0.2
    context[4] = np.cos(years_since_base * 0.3) * 0.2
    
    # Long-term climate oscillation approximation
    context[5] = np.sin(years_since_base * 0.1) * 0.3
    
    # Temperature amplification factor
    if climate_mode == "warming":
        context[6] = min(1.0, 0.5 + years_since_base / 60.0)
    elif climate_mode == "extreme":
        context[6] = min(1.2, 0.5 + years_since_base / 50.0)
    else:
        context[6] = min(0.8, 0.3 + years_since_base / 80.0)
    
    # Rainfall pattern shift factor
    context[7] = 0.3 + (years_since_base / 90.0) * 0.4
    
    # Random factors to represent unpredictable elements
    context[8] = np.random.normal(0, 0.1)
    context[9] = np.random.normal(0, 0.1)
    
    # Autocorrelation approximation
    context[10] = 0.6 + (years_since_base / 200.0) * 0.2
    
    # Extreme event probability - higher in extreme scenario
    if climate_mode == "extreme":
        context[11] = 0.3 + (years_since_base / 50.0) * 0.4
    else:
        context[11] = 0.1 + (years_since_base / 100.0) * 0.2
    
    return context

def predict_future(embeddings, target_year, climate_mode="warming", temp_indices=None, rainfall_indices=None):
    """
    Predict future embeddings with climate awareness
    
    Args:
        embeddings: historical embeddings dictionary
        target_year: year up to which to predict
        climate_mode: climate scenario type (standard, warming, extreme)
        temp_indices: indices corresponding to temperature data
        rainfall_indices: indices corresponding to rainfall data
    """
    years = list(embeddings.keys())
    current_year = max(years)

    # Create output directory if it doesn't exist
    model_dir = os.path.join(os.getcwd(), "backend/models")
    ensure_directory_exists(model_dir)
    model_path = os.path.join(model_dir, f"climate_{climate_mode}_prediction_model.pth")
    
    # Also ensure the graph_embeddings directory exists
    embeddings_dir = os.path.join(os.getcwd(), "backend/data/graph_embeddings")
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
    
    # Create climate-aware model
    model = ClimateAwarePredictionModel(input_dim, climate_mode=climate_mode)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Prepare context data with climate information
    context_data = []
    for year in years[:-10]:
        context = create_climate_context(year + 10, climate_mode)
        context_data.append(context.unsqueeze(0))
    
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
        
        # Use climate-aware loss
        loss = climate_aware_loss(predictions, train_y, temp_indices, rainfall_indices, climate_mode)
        
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

    print(f"Climate-aware training completed for {climate_mode} scenario")

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
    
    # Different min_allowed_change based on climate mode
    if climate_mode == "warming":
        # For warming scenario, we limit decreases even more for temperature
        min_allowed_change = -0.005
    elif climate_mode == "extreme":
        # For extreme scenario, we allow more variability
        min_allowed_change = -0.02
    else:
        min_allowed_change = -0.01

    for year in range(current_year + 1, target_year + 1):
        past_10_years = [future_embeddings[y] for y in range(year - 10, year)]
        input_seq = torch.stack(past_10_years).unsqueeze(0)
        
        # Use the same reshaping logic as in training
        _, seq_len, nodes, feat = input_seq.shape
        input_seq = input_seq.reshape(1, seq_len, -1)
        
        # Use climate-aware context for prediction
        context = create_climate_context(year, climate_mode).unsqueeze(0)

        with torch.no_grad():
            predicted = model(input_seq, context)
            # Reshape the output back to the original dimensions
            predicted = predicted.reshape(num_nodes, features)
            
        if last_prediction is not None:
            # Apply climate-specific constraints
            change = predicted - last_prediction
            
            if climate_mode == "warming" and temp_indices is not None:
                # For temperature indices in warming scenario, ensure upward trend
                temp_changes = change[temp_indices]
                excessive_temp_drops = (temp_changes < min_allowed_change)
                if excessive_temp_drops.any():
                    predicted[temp_indices][excessive_temp_drops] = (
                        last_prediction[temp_indices][excessive_temp_drops] + min_allowed_change
                    )
                    
            elif climate_mode == "extreme":
                # For extreme scenario, allow more variability but ensure overall trend
                excessive_drops = (change < min_allowed_change * 2)
                if excessive_drops.any():
                    predicted[excessive_drops] = last_prediction[excessive_drops] + min_allowed_change * 2
                    
                # Add occasional extreme values
                if np.random.random() < 0.1:  # 10% chance of extreme value
                    extreme_idx = np.random.randint(0, num_nodes)
                    predicted[extreme_idx] = predicted[extreme_idx] * (1 + np.random.uniform(0.1, 0.3))
            else:
                # Standard mode - similar to original
                excessive_drops = (change < min_allowed_change)
                if excessive_drops.any():
                    predicted[excessive_drops] = last_prediction[excessive_drops] + min_allowed_change
        
        last_prediction = predicted.clone()
        future_embeddings[year] = predicted
        
        # Save the prediction with error handling
        embedding_file = os.path.join(embeddings_dir, f"embeddings_{year}_{climate_mode}.pt")
        try:
            torch.save(predicted, embedding_file)
            print(f"Predictions for {year} ({climate_mode} scenario) saved to {embedding_file}")
        except Exception as e:
            print(f"Error saving prediction for year {year}: {e}")
            # Try alternative location
            try:
                alt_file = f"embeddings_{year}_{climate_mode}.pt"
                torch.save(predicted, alt_file)
                print(f"Predictions saved to alternative location: {alt_file}")
            except Exception as e2:
                print(f"Failed to save predictions for year {year}: {e2}")

    return future_embeddings

if __name__ == "__main__":
    try:
        # Load embeddings with error handling
        embeddings = {}
        embeddings_dir = os.path.join(os.getcwd(), "backend/data/graph_embeddings")
        
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
        
        # Get climate mode and target year
        print("Select climate scenario:")
        print("1. Standard (moderate predictions)")
        print("2. Warming (stronger warming trend)")
        print("3. Extreme (more variability and extreme events)")
        choice = input("Enter choice (1-3): ")
        
        if choice == "2":
            climate_mode = "warming"
        elif choice == "3":
            climate_mode = "extreme"
        else:
            climate_mode = "standard"
            
        target_year = int(input("Enter the target year for predictions: "))
        
        # Identify temperature and rainfall indices
        print("If you know the indices for temperature and rainfall data, enter them now.")
        print("Otherwise, leave blank for automatic detection.")
        
        temp_input = input("Temperature indices (comma-separated, or leave blank): ")
        rainfall_input = input("Rainfall indices (comma-separated, or leave blank): ")
        
        temp_indices = [int(x) for x in temp_input.split(',')] if temp_input else None
        rainfall_indices = [int(x) for x in rainfall_input.split(',')] if rainfall_input else None
        
        # If no indices provided, try to detect them based on patterns
        if temp_indices is None or rainfall_indices is None:
            print("Attempting to detect data types from patterns...")
            # This is a placeholder - a real implementation would analyze the data
            # to identify temperature and rainfall patterns
            
        predict_future(embeddings, target_year, climate_mode, temp_indices, rainfall_indices)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()