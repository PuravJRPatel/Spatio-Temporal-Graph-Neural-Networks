import torch
import matplotlib.pyplot as plt
import numpy as np
from lstm_prediction import ClimateAwarePredictionModel

# Load the trained model (Replace 'model.pth' with your actual saved model path)
model = ClimateAwarePredictionModel()
model.load_state_dict(torch.load('backend\models\climate_standard_prediction_model.pth'))
model.eval()

# Define the years for prediction (adjust range as needed)
years = np.arange(2024, 2035)

# Placeholder input data (replace with real test data if available)
# Ensure the shape matches the expected input format of your model
input_data = torch.randn(len(years), model.input_size)  # Adjust dimensions

# Generate predictions with the trained model
with torch.no_grad():
    predictions = model(input_data)

# Extract rainfall and temperature predictions
predicted_rainfall = predictions[:, 0].numpy()
predicted_temperature = predictions[:, 1].numpy()

# Visualization
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(years, predicted_rainfall, 'b-o', label='Predicted Rainfall')
ax2.plot(years, predicted_temperature, 'r-s', label='Predicted Temperature')

ax1.set_xlabel('Year')
ax1.set_ylabel('Rainfall (normalized)', color='blue')
ax2.set_ylabel('Temperature (normalized)', color='red')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Climate Prediction Model Results')
plt.show()
