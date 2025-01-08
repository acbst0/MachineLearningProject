import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import joblib
import threading
import os

# Load dataset
file_path = "winequality-white.csv"
df = pd.read_csv(file_path, sep=";")

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tkinter application
root = tk.Tk()
root.title("Machine Learning Interface")
root.state('zoomed')  # Fullscreen mode

# Main frame
main_frame = tk.Frame(root, bg="white")
main_frame.pack(fill=tk.BOTH, expand=True)

# Left frame (for plots)
left_frame = tk.Frame(main_frame, bg="white", width=800, height=500)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Middle frame (for metrics)
middle_frame = tk.Frame(main_frame, bg="white", width=800, height=100)
middle_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# Right frame (for dataset and table)
right_frame = tk.Frame(main_frame, bg="white", width=800, height=500)
right_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

# Bottom frame (for buttons)
bottom_frame = tk.Frame(main_frame, bg="#d9d9d9", height=100)
bottom_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

# Placeholder for plots
fig = Figure(figsize=(10, 8), dpi=100)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title("Model Performance Plot", fontsize=12)
ax2.set_title("Additional Performance Details", fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)

canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

# Treeview for dataset display
tree = ttk.Treeview(right_frame, columns=list(df.columns), show="headings", height=20)
for col in df.columns:
    tree.heading(col, text=col)
    tree.column(col, anchor=tk.CENTER, width=100)

for index, row in df.iterrows():
    tree.insert("", tk.END, values=list(row))

tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Label to display model results
result_label = tk.Label(middle_frame, text="Model Results Will Be Shown Here", font=("Arial", 12), bg="white", wraplength=600, justify="left")
result_label.pack(pady=10, padx=10, anchor="w")

# Progress bar
progress = ttk.Progressbar(bottom_frame, orient=tk.HORIZONTAL, length=200, mode='indeterminate')

# Output directory for saving models and logs
output_dir = "model_outputs"
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "model_performance.log")

# Log function
def log_results(result_text):
    with open(log_file_path, "a") as log_file:
        log_file.write(result_text + "\n\n")

def save_and_notify(filename, folder):
    out = output_dir+"/"+folder
    os.makedirs(out, exist_ok=True)
    filepath = os.path.join(out, filename)
    fig.savefig(filepath)
    messagebox.showinfo("Saved", f"Graphic also saved as {filename}")

# Functions to save/load models
def save_model(model, filename, name):
    out = output_dir+"/"+name 
    filepath = os.path.join(out, filename)
    joblib.dump(model, filepath)

def load_model(filename):
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None

def open_test_window():
    test_window = tk.Toplevel(root)
    test_window.title("Prediction Window")
    test_window.geometry("400x600")

    input_frame = tk.Frame(test_window, bg="white")
    input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    result_frame = tk.Frame(test_window, bg="#f0f0f0")
    result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    inputs = {}
    for i, col in enumerate(X.columns):
        label = tk.Label(input_frame, text=col, font=("Arial", 10), bg="white")
        label.grid(row=i, column=0, padx=5, pady=5)
        entry = tk.Entry(input_frame)
        entry.grid(row=i, column=1, padx=5, pady=5)
        inputs[col] = entry

    # Label to display prediction result
    result_label = tk.Label(result_frame, text="Prediction result will appear here.", font=("Arial", 12), bg="#f0f0f0", wraplength=600)
    result_label.pack(pady=10)

    def predict():
        model_name = model_selection.get()
        if model_name == "SVR":
            model_name = "SVR/"+model_name
        elif model_name == "DecisionTree":
            model_name = "DecisionTree/"+model_name
        elif model_name == "RandomForest":
            model_name = "RFT/"+model_name
        model = load_model(f"{model_name}.joblib")
        if model is None:
            messagebox.showerror("Error", f"Model '{model_name}' could not be loaded.")
            return

        try:
            test_data = pd.DataFrame([[float(inputs[col].get()) for col in X.columns]], columns=X.columns)
            prediction = model.predict(test_data)
            result_label.config(text=f"Prediction: {prediction[0]:.4f}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

    model_selection = ttk.Combobox(test_window, values=["SVR", "DecisionTree", "RandomForest"], state="readonly")
    model_selection.set("Select Model")
    model_selection.pack(pady=10)

    predict_button = tk.Button(test_window, text="Predict", command=predict, bg="#0078d4", fg="white")
    predict_button.pack(pady=10)

# Compute additional metric: Mean Percentage Error (MPE)
def mean_percentage_error(y_true, y_pred):
    return np.mean((y_true - y_pred) / y_true) * 100

# Model functions
def start_training(func):
    def wrapper():
        progress.pack(side=tk.RIGHT, padx=10, pady=10)
        progress.start()
        threading.Thread(target=lambda: [func(), progress.stop(), progress.pack_forget()]).start()
    return wrapper

@start_training
def apply_svr():
    out = output_dir + "/SVR"
    os.makedirs(out, exist_ok=True)
    model_filename = "SVR.joblib"
    model_path = os.path.join(out, model_filename)
    model = load_model(model_path)
    if model is None:
        model = SVR(kernel="linear", C=1.0)
        model.fit(X_train, y_train)
        save_model(model, model_filename, "SVR")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mpe = mean_percentage_error(y_test, y_pred)

    result_text = (
        f"Model: SVR\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Mean Squared Error (MSE): {mse:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"R2 Score: {r2:.4f}\n"
        f"Mean Percentage Error (MPE): {mpe:.4f}"
    )
    result_label.config(text=result_text)
    log_results(result_text)

    ax1.clear()
    ax1.scatter(range(len(y_test)), y_test, label="True Values", color="blue", alpha=0.6)
    ax1.scatter(range(len(y_pred)), y_pred, label="Predictions", color="red", alpha=0.6)
    ax1.legend(fontsize=10)

    ax2.clear()
    ax2.plot(y_test.values - y_pred, label="Prediction Error", color="green")
    ax2.legend(fontsize=10)

    canvas.draw()  # Grafiklerin çizimini tamamla
    save_and_notify("svr_graph.png", "SVR")  # Grafik kaydet

@start_training
def apply_decision_tree():
    out = output_dir + "/DecisionTree"
    os.makedirs(out, exist_ok=True)
    model_filename = "DecisionTree.joblib"
    model_path = os.path.join(out, model_filename)
    model = load_model(model_path)
    if model is None:
        param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        save_model(model, model_filename, "DecisionTree")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mpe = mean_percentage_error(y_test, y_pred)
    #crvs = cross_val_score(DecisionTreeRegressor(random_state=42), X_train, y_train, cv=5)


    result_text = (
        f"Model: Decision Tree\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Mean Squared Error (MSE): {mse:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"R2 Score: {r2:.4f}\n"
        f"Mean Percentage Error (MPE): {mpe:.4f}"
        #f"Cross-Validation Score (CVS): {crvs:.4f}"
    )
    result_label.config(text=result_text)
    log_results(result_text)

    ax1.clear()
    plot_tree(model, ax=ax1, feature_names=X.columns, filled=True)

    ax2.clear()
    ax2.hist(y_test.values - y_pred, bins=20, color="orange", alpha=0.7, label="Prediction Error Distribution")
    ax2.legend(fontsize=10)

    canvas.draw()
    save_and_notify("decision_tree_graph.png", "DecisionTree")

@start_training
def apply_random_forest():
    out = output_dir + "/RFT"
    os.makedirs(out, exist_ok=True)
    model_filename = "RandomForest.joblib"
    model_path = os.path.join(out, model_filename)
    model = load_model(model_path)
    if model is None:
        param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)  # Daha az katlama
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        save_model(model, model_filename, "RFT")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mpe = mean_percentage_error(y_test, y_pred)

    result_text = (
        f"Model: Random Forest\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Mean Squared Error (MSE): {mse:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"R2 Score: {r2:.4f}\n"
        f"Mean Percentage Error (MPE): {mpe:.4f}"
    )
    result_label.config(text=result_text)
    log_results(result_text)

    ax1.clear()
    for i, estimator in enumerate(model.estimators_):
        if i < 3:  # Sadece ilk 3 ağacı çizelim
            plot_tree(estimator, ax=ax1, feature_names=X.columns, filled=True)
            ax1.set_title(f"Random Forest Tree {i+1}")
            canvas.draw()  # Her ağaç için çizimi güncelle
            ax2.clear()
            ax2.hist(y_test.values - y_pred, bins=20, color="orange", alpha=0.7, label="Prediction Error Distribution")
            ax2.legend(fontsize=10)
            canvas.draw()  # Son grafiği çiz
            save_and_notify(f"random_forest_tree_{i + 1}.png", "RFT")  # Her ağaç için kaydet

    canvas.draw()  # Son grafiği çiz
    save_and_notify("random_forest_graph.png", "RFT")

# Buttons
button1 = tk.Button(bottom_frame, text="SVR", font=("Arial", 10), bg="#0078d4", fg="white", command=apply_svr)
button2 = tk.Button(bottom_frame, text="Decision Tree", font=("Arial", 10), bg="#0078d4", fg="white", command=apply_decision_tree)
button3 = tk.Button(bottom_frame, text="Random Forest", font=("Arial", 10), bg="#0078d4", fg="white", command=apply_random_forest)
button4 = tk.Button(bottom_frame, text="Test", font=("Arial", 10), bg="#00b300", fg="white", command=open_test_window)

button1.pack(side=tk.LEFT, padx=10, pady=10)
button2.pack(side=tk.LEFT, padx=10, pady=10)
button3.pack(side=tk.LEFT, padx=10, pady=10)
button4.pack(side=tk.LEFT, padx=10, pady=10)

# Configure grid weight
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_rowconfigure(1, weight=0)
main_frame.grid_rowconfigure(2, weight=0)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)

# Main loop
root.mainloop()
