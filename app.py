import os
import torch
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(actual_data, predicted_data):
    """
    Visualize rainfall and temperature predictions using Streamlit.
    
    Args:
    - actual_data (dict): Dictionary of actual embeddings/data
    - predicted_data (dict): Dictionary of predicted embeddings/data
    """
    # Extract years
    all_years = sorted(list(actual_data.keys()) + list(predicted_data.keys()))
    start_year = min(all_years)
    end_year = max(all_years)
    
    # Separate rainfall and temperature predictions
    def extract_component(data, component_index):
        return [data[year][:, component_index].mean().item() for year in sorted(data.keys())]
    
    actual_rainfall = extract_component(actual_data, 0)
    actual_temperature = extract_component(actual_data, 1)
    
    predicted_rainfall = extract_component(predicted_data, 0)
    predicted_temperature = extract_component(predicted_data, 1)
    
    # Prepare years for plotting
    actual_years = sorted(actual_data.keys())
    predicted_years = sorted(predicted_data.keys())
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Rainfall and Temperature Predictions', fontsize=16)
    
    # Rainfall Subplot
    ax1.plot(actual_years, actual_rainfall, 'bo-', label='Actual Rainfall')
    ax1.plot(predicted_years, predicted_rainfall, 'ro--', label='Predicted Rainfall')
    ax1.set_title('Rainfall Prediction')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Rainfall (mm)')
    ax1.legend()
    ax1.grid(True)
    
    # Temperature Subplot
    ax2.plot(actual_years, actual_temperature, 'bo-', label='Actual Temperature')
    ax2.plot(predicted_years, predicted_temperature, 'ro--', label='Predicted Temperature')
    ax2.set_title('Temperature Prediction')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Temperature (°C)')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and display in Streamlit
    plt.tight_layout()
    st.pyplot(fig)

def load_data(target_year):
    """
    Load the actual and predicted embeddings for the specified range of years.
    
    Args:
    - target_year (int): The target year for which predictions should be visualized.
    
    Returns:
    - actual_embeddings (dict): Dictionary of actual data embeddings for years 1980-2023.
    - predicted_embeddings (dict): Dictionary of predicted data embeddings for years 2024 to target_year.
    """
    actual_embeddings = {}
    predicted_embeddings = {}
    
    # Load actual embeddings for years 1980-2023
    for year in range(1980, 2024):
        actual_embeddings[year] = torch.load(os.path.join("backend/data/graph_embeddings", f"embeddings_{year}.pt"))
    
    # Load predicted embeddings for years 2024 to target_year
    for year in range(2024, target_year + 1):
        predicted_embeddings[year] = torch.load(os.path.join("backend/data/graph_embeddings", f"embeddings_{year}.pt"))
    
    return actual_embeddings, predicted_embeddings

def main():
    st.title("Climate Predictions: Rainfall and Temperature")
    
    # Input for target year
    target_year = st.number_input("Enter the target year for visualization", min_value=2024, max_value=2050, value=2024)
    
    # Load the embeddings based on the selected year
    actual_embeddings, predicted_embeddings = load_data(target_year)
    
    # Visualize predictions
    visualize_predictions(actual_embeddings, predicted_embeddings)

if __name__ == "__main__":
    main()

