import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import threading
import os

# Veri setini yükleme
file_path = "winequality-white.csv"
df = pd.read_csv(file_path, sep=";")

# Girdiler ve hedef
X = df.drop("quality", axis=1)
y = df["quality"]

# Eğitim ve test seti oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tkinter uygulaması
root = tk.Tk()
root.title("Makine Öğrenimi Arayüzü")
root.geometry("1400x600")

# Ana çerçeve
main_frame = tk.Frame(root, bg="white")
main_frame.pack(fill=tk.BOTH, expand=True)

# Sol çerçeve (Grafik için)
left_frame = tk.Frame(main_frame, bg="white", width=700, height=500)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Sağ çerçeve (Dataset için)
right_frame = tk.Frame(main_frame, bg="white", width=700, height=500)
right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Alt çerçeve (Butonlar için)
bottom_frame = tk.Frame(main_frame, bg="#d9d9d9", height=100)
bottom_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

# Grafik için placeholder
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Model Performans Grafiği")
canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

# Veri seti gösterimi için Treeview
tree = ttk.Treeview(right_frame, columns=list(df.columns), show="headings", height=20)
for col in df.columns:
    tree.heading(col, text=col)
    tree.column(col, anchor=tk.CENTER, width=100)

for index, row in df.iterrows():
    tree.insert("", tk.END, values=list(row))

tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Model sonuçlarını göstermek için etiket
result_label = tk.Label(left_frame, text="Model Sonuçları Burada Gösterilecek", font=("Arial", 12), bg="white", wraplength=500)
result_label.pack(pady=10)

# Yükleme çubuğu
progress = ttk.Progressbar(bottom_frame, orient=tk.HORIZONTAL, length=200, mode='indeterminate')

# Kaydetme ve mesaj kutusu
output_dir = "model_outputs"
os.makedirs(output_dir, exist_ok=True)

def save_and_notify(filename):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    messagebox.showinfo("Saved", f"Graphic also saved as {filename}")

def start_training(func):
    def wrapper():
        progress.pack(side=tk.RIGHT, padx=10, pady=10)
        progress.start()
        threading.Thread(target=lambda: [func(), progress.stop(), progress.pack_forget()]).start()
    return wrapper

# Model fonksiyonları
@start_training
def apply_svr():
    model = SVR(kernel="linear")
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    mean_cv_score = np.mean(cv_scores)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Hata oranları
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Sonuçları göster
    result_text = (
        f"Model: SVR\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Mean Squared Error (MSE): {mse:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"R2 Score: {r2:.4f}\n"
        f"Cross-Validation R2 (Mean): {mean_cv_score:.4f}"
    )
    result_label.config(text=result_text)

    # Grafik güncellemesi
    ax.clear()
    ax.scatter(range(len(y_test)), y_test, label="Gerçek Değerler", color="blue", alpha=0.6)
    ax.scatter(range(len(y_pred)), y_pred, label="Tahminler", color="red", alpha=0.6)
    ax.set_title("SVR Performansı (Scatter Plot)")
    ax.legend()
    canvas.draw()
    save_and_notify("svr_performance.png")

@start_training
def apply_decision_tree():
    model = DecisionTreeRegressor(random_state=42)
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    mean_cv_score = np.mean(cv_scores)

    model.fit(X_train, y_train)

    # Hata oranları
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Sonuçları göster
    result_text = (
        f"Model: Decision Tree\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Mean Squared Error (MSE): {mse:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"R2 Score: {r2:.4f}\n"
        f"Cross-Validation R2 (Mean): {mean_cv_score:.4f}"
    )
    result_label.config(text=result_text)

    # Grafik güncellemesi
    ax.clear()
    plot_tree(model, ax=ax, feature_names=X.columns, filled=True)
    ax.set_title("Decision Tree")
    canvas.draw()
    save_and_notify("decision_tree.png")

@start_training
def apply_random_forest():
    model = RandomForestRegressor(random_state=42, n_estimators=5)
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    mean_cv_score = np.mean(cv_scores)

    model.fit(X_train, y_train)

    # Hata oranları
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Sonuçları göster
    result_text = (
        f"Model: Random Forest\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Mean Squared Error (MSE): {mse:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"R2 Score: {r2:.4f}\n"
        f"Cross-Validation R2 (Mean): {mean_cv_score:.4f}"
    )
    result_label.config(text=result_text)

    # Grafik güncellemesi
    ax.clear()
    for i, estimator in enumerate(model.estimators_):
        if i < 3:  # Sadece ilk 3 ağacı çizelim
            plot_tree(estimator, ax=ax, feature_names=X.columns, filled=True)
            ax.set_title(f"Random Forest Tree {i+1}")
            canvas.draw()
            save_and_notify(f"random_forest_tree_{i+1}.png")

# Butonlar
button1 = tk.Button(bottom_frame, text="SVR", font=("Arial", 10), bg="#0078d4", fg="white", command=apply_svr)
button2 = tk.Button(bottom_frame, text="Decision Tree", font=("Arial", 10), bg="#0078d4", fg="white", command=apply_decision_tree)
button3 = tk.Button(bottom_frame, text="Random Forest", font=("Arial", 10), bg="#0078d4", fg="white", command=apply_random_forest)

button1.pack(side=tk.LEFT, padx=10, pady=10)
button2.pack(side=tk.LEFT, padx=10, pady=10)
button3.pack(side=tk.LEFT, padx=10, pady=10)

# Grid ağı yapılandırma
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)

# Ana döngü
root.mainloop()
