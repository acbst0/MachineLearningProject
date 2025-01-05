# Machine Learning Model GUI for White Wine Quality Dataset

## Overview
This project features a graphical user interface (GUI) for experimenting with machine learning models on the **White Wine Quality** dataset. The application allows users to train, evaluate, and visualize the performance of machine learning algorithms directly from the interface.

## Features
- **Built-in Models**:
  - Support Vector Regression (SVR)
  - Decision Tree Regressor
  - Random Forest Regressor
- **Metrics Displayed**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R2 Score
  - Cross-Validation R2 (Mean)
- **Visualization**:
  - SVR predictions vs. actual values shown as scatter plots.
  - Decision Tree and Random Forest tree structures visualized graphically.
- **Save Output**:
  - All visualizations are automatically saved in the `model_outputs` directory.

## Requirements
Ensure you have Python installed and the following libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tkinter` (comes pre-installed with Python)

## Install all dependencies using:
''bash''
pip install -r requirements.txt

## Dataset Link
  https://archive.ics.uci.edu/dataset/186/wine+quality
