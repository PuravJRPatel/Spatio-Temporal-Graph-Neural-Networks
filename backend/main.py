import os
import sys
import argparse
import torch

# Import functions from your existing scripts
from scripts.build_graph import load_data, build_graph, save_graph
from scripts.train_gcn import GCN, train_model, save_embeddings
from scripts.lstm_prediction import predict_future, AdvancedPredictionModel

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

def process_climate_data(data_path, year_range=None):
    """Load and process climate data, build graphs for each year."""
    print("Step 1: Processing climate data and building graphs...")
    
    # Setup directories
    output_dir = "backend/data/processed_graphs"
    ensure_directory_exists(output_dir)
    
    # Load the data
    df = load_data(data_path)
    
    # Determine which years to process
    if year_range:
        start_year, end_year = year_range
        years = [year for year in df['year'].unique() if start_year <= year <= end_year]
    else:
        years = df['year'].unique()
    
    # Process each year
    for year in years:
        print(f"  Processing year {year}...")
        df_year = df[df['year'] == year]
        graph = build_graph(df_year)
        save_graph(graph, year, output_dir)
    
    print("Graph processing complete")
    return years

def train_gcn_models(years, config):
    """Train GCN models for each year's graph and save embeddings."""
    print("Step 2: Training GCN models and generating embeddings...")
    
    # Setup directories
    graph_dir = "backend/data/processed_graphs"
    embeddings_dir = "backend/data/graph_embeddings"
    ensure_directory_exists(embeddings_dir)
    
    # Train model for each year
    for year in years:
        print(f"\nTraining GCN for year {year}")
        try:
            # Load the graph
            graph_path = os.path.join(graph_dir, f"graph_{year}.pt")
            if not os.path.exists(graph_path):
                print(f"  Warning: Graph file for year {year} not found. Skipping.")
                continue
            
            graph = torch.load(graph_path)
            
            # Train the model
            model = GCN(in_channels=graph.x.shape[1], 
                       hidden_channels=config['hidden_dim'], 
                       out_channels=config['output_dim'])
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            
            for epoch in range(config['num_epochs']):
                model.train()
                optimizer.zero_grad()
                output = model(graph.x, graph.edge_index)
                loss = torch.nn.functional.mse_loss(output, graph.x)
                loss.backward()
                optimizer.step()
                
                if epoch % 100 == 0:
                    print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
            
            # Save the embeddings
            save_embeddings(output, year, embeddings_dir)
            
        except Exception as e:
            print(f"  Error processing year {year}: {e}")
    
    print("GCN training complete")

def make_future_predictions(target_year):
    """Make predictions for future years using trained models."""
    print(f"Step 3: Making predictions up to year {target_year}...")
    
    # Load embeddings
    embeddings = {}
    embeddings_dir = os.path.join(os.getcwd(), "backend/data/graph_embeddings")
    
    if not os.path.exists(embeddings_dir):
        print(f"Error: Directory {embeddings_dir} does not exist.")
        return None
    
    # Load available embeddings
    missing_years = []
    available_years = []
    for year in range(1980, 2024):  # Assuming this is your data range
        embedding_file = os.path.join(embeddings_dir, f"embeddings_{year}.pt")
        if os.path.exists(embedding_file):
            try:
                embeddings[year] = torch.load(embedding_file)
                available_years.append(year)
            except Exception as e:
                print(f"Error loading embeddings for year {year}: {e}")
                missing_years.append(year)
        else:
            missing_years.append(year)
    
    if missing_years:
        print(f"Warning: Embeddings missing for years: {missing_years}")
        if len(missing_years) > 10:
            print("Too many missing embeddings. Please check your data.")
            return None
    
    # Make predictions
    future_embeddings = predict_future(embeddings, target_year)
    print(f"Predictions complete up to year {target_year}")
    
    return future_embeddings

def main():
    """Main function to orchestrate the entire climate prediction pipeline."""
    parser = argparse.ArgumentParser(description="Climate Prediction Pipeline")
    parser.add_argument("--data", type=str, default="backend/data/Processed Data/climate_data.csv",
                        help="Path to the climate data CSV file")
    parser.add_argument("--years", type=str, default="1980-2023",
                        help="Year range to process (format: start-end)")
    parser.add_argument("--target", type=int, default=2030,
                        help="Target year for future predictions")
    parser.add_argument("--skip-graphs", action="store_true",
                        help="Skip graph building step")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip GCN training step")
    parser.add_argument("--skip-predict", action="store_true",
                        help="Skip future prediction step")
    
    args = parser.parse_args()
    
    # Parse year range
    year_range = None
    if args.years:
        try:
            start, end = map(int, args.years.split('-'))
            year_range = (start, end)
        except ValueError:
            print(f"Invalid year range format: {args.years}. Using all available years.")
    
    # GCN training configuration
    gcn_config = {
        'hidden_dim': 64,
        'output_dim': 2,
        'num_epochs': 1000,
        'learning_rate': 0.001,
    }
    
    # Run the pipeline
    years = None
    
    # Step 1: Process climate data and build graphs
    if not args.skip_graphs:
        years = process_climate_data(args.data, year_range)
    
    # Step 2: Train GCN models
    if not args.skip_train:
        if years is None:
            # If we skipped graph building, determine years from available graph files
            graph_dir = "backend/data/processed_graphs"
            if os.path.exists(graph_dir):
                years = []
                for filename in os.listdir(graph_dir):
                    if filename.startswith("graph_") and filename.endswith(".pt"):
                        try:
                            year = int(filename[6:-3])  # Extract year from "graph_YYYY.pt"
                            years.append(year)
                        except ValueError:
                            continue
                years.sort()
            
            if not years:
                print("No graph files found. Please run the graph building step first.")
                return
        
        train_gcn_models(years, gcn_config)
    
    # Step 3: Make future predictions
    if not args.skip_predict:
        future_embeddings = make_future_predictions(args.target)
        if future_embeddings:
            print(f"Successfully generated predictions up to year {args.target}")
    
    print("Climate prediction pipeline completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()