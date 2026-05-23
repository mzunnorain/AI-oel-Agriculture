"""
Tkinter GUI for Smart Agriculture Decision Support System
Integrates Decision Tree Classifier, KMeans Clustering, and Linear Regression models
"""
import sys
import os

# Allow running directly from src/ or from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

from models import ModelFactory, get_cluster_guidance
from preprocessing import DataPreprocessor
from utils import create_directories, format_metrics_display, validate_input, FEATURE_RANGES


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG_COLOR = '#f5f5f5'
ACCENT = '#2e7d32'       # dark green
ACCENT_LIGHT = '#e8f5e9'
HEADER_FONT = ('Arial', 13, 'bold')
LABEL_FONT = ('Arial', 10)
MONO_FONT = ('Courier New', 10)


class AgricultureGUI:
    """Main GUI Application for Agriculture Decision Support System."""

    def __init__(self, root):
        self.root = root
        self.root.title("Smart Agriculture Decision Support System")
        self.root.geometry("1280x820")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(True, True)

        # Core components
        self.model_factory = ModelFactory()
        self.preprocessor = DataPreprocessor()
        self.current_data = None
        self._canvas_widget = None  # track embedded matplotlib canvas

        create_directories()
        self._setup_styles()
        self._setup_ui()

    # ------------------------------------------------------------------
    # Style configuration
    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=LABEL_FONT, padding=[10, 4])
        style.configure('Accent.TButton', foreground='white', background=ACCENT,
                        font=('Arial', 10, 'bold'))
        style.map('Accent.TButton', background=[('active', '#1b5e20')])

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _setup_ui(self):
        # Header bar
        header = tk.Frame(self.root, bg=ACCENT, height=50)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="🌾  Smart Agriculture Decision Support System",
            bg=ACCENT, fg='white',
            font=('Arial', 14, 'bold'),
        ).pack(side=tk.LEFT, padx=15, pady=10)

        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_prediction = ttk.Frame(self.notebook)
        self.tab_metrics = ttk.Frame(self.notebook)
        self.tab_visualization = ttk.Frame(self.notebook)
        self.tab_about = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_prediction,   text="  Prediction  ")
        self.notebook.add(self.tab_metrics,      text="  Model Metrics  ")
        self.notebook.add(self.tab_visualization, text="  Visualizations  ")
        self.notebook.add(self.tab_about,        text="  About  ")

        self._setup_prediction_tab()
        self._setup_metrics_tab()
        self._setup_visualization_tab()
        self._setup_about_tab()

    # ------------------------------------------------------------------
    # Prediction tab
    # ------------------------------------------------------------------
    def _setup_prediction_tab(self):
        # Instructions
        info = ttk.Label(
            self.tab_prediction,
            text="Enter soil and climatic parameters, then click Predict.",
            font=LABEL_FONT,
        )
        info.pack(anchor=tk.W, padx=12, pady=(8, 2))

        # Input grid
        input_frame = ttk.LabelFrame(
            self.tab_prediction, text="Input Parameters", padding=12
        )
        input_frame.pack(fill=tk.X, padx=12, pady=6)

        self.input_fields = {}
        # 7 agronomic features — Soil_Type removed (not in dataset)
        input_params = [
            ('Nitrogen',    'mg/kg',  '0 – 140'),
            ('Phosphorus',  'mg/kg',  '5 – 145'),
            ('Potassium',   'mg/kg',  '5 – 205'),
            ('Temperature', '°C',     '8.8 – 43.7'),
            ('Humidity',    '%',      '14 – 99'),
            ('pH',          '',       '3.5 – 9.9'),
            ('Rainfall',    'mm',     '20 – 298'),
        ]

        for i, (param, unit, rng) in enumerate(input_params):
            row, col = divmod(i, 4)
            lbl_text = f"{param}" + (f" ({unit})" if unit else "")
            ttk.Label(input_frame, text=lbl_text, font=LABEL_FONT).grid(
                row=row * 2, column=col, sticky=tk.W, padx=8, pady=(6, 0)
            )
            ttk.Label(input_frame, text=f"Range: {rng}",
                      font=('Arial', 8), foreground='grey').grid(
                row=row * 2 + 1, column=col, sticky=tk.W, padx=8, pady=(0, 4)
            )
            entry = ttk.Entry(input_frame, width=14)
            entry.grid(row=row * 2, column=col + 4, rowspan=2,
                       padx=8, pady=4, sticky=tk.W)
            self.input_fields[param] = entry

        # Buttons
        btn_frame = ttk.Frame(self.tab_prediction)
        btn_frame.pack(fill=tk.X, padx=12, pady=6)

        ttk.Button(btn_frame, text="Load Models",
                   command=self._load_models, style='Accent.TButton').pack(
            side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Predict",
                   command=self._make_prediction, style='Accent.TButton').pack(
            side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Clear",
                   command=self._clear_inputs).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Fill Example",
                   command=self._fill_example).pack(side=tk.LEFT, padx=4)

        # Output area
        out_frame = ttk.LabelFrame(
            self.tab_prediction, text="Integrated Predictions", padding=10
        )
        out_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        sb = ttk.Scrollbar(out_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text = tk.Text(
            out_frame, height=14, font=MONO_FONT,
            yscrollcommand=sb.set, bg='#fafafa', relief=tk.FLAT
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.output_text.yview)

        # Status bar
        self.status_var = tk.StringVar(value="Ready — load models to begin.")
        ttk.Label(self.tab_prediction, textvariable=self.status_var,
                  foreground='grey', font=('Arial', 9)).pack(
            anchor=tk.W, padx=12, pady=(0, 4))

    # ------------------------------------------------------------------
    # Metrics tab
    # ------------------------------------------------------------------
    def _setup_metrics_tab(self):
        ttk.Label(self.tab_metrics, text="Model Performance Metrics",
                  font=HEADER_FONT).pack(padx=12, pady=10)

        sb = ttk.Scrollbar(self.tab_metrics)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.metrics_text = tk.Text(
            self.tab_metrics, font=MONO_FONT,
            yscrollcommand=sb.set, bg='#fafafa', relief=tk.FLAT
        )
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        sb.config(command=self.metrics_text.yview)

        ttk.Button(self.tab_metrics, text="Refresh Metrics",
                   command=self._display_metrics,
                   style='Accent.TButton').pack(pady=6)

    # ------------------------------------------------------------------
    # Visualization tab
    # ------------------------------------------------------------------
    def _setup_visualization_tab(self):
        btn_frame = ttk.Frame(self.tab_visualization)
        btn_frame.pack(fill=tk.X, padx=12, pady=8)

        ttk.Button(btn_frame, text="Feature Importance",
                   command=self._plot_feature_importance,
                   style='Accent.TButton').pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cluster Scatter",
                   command=self._plot_clusters,
                   style='Accent.TButton').pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Residual Plot",
                   command=self._plot_residuals,
                   style='Accent.TButton').pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Actual vs Predicted",
                   command=self._plot_actual_vs_predicted,
                   style='Accent.TButton').pack(side=tk.LEFT, padx=4)

        self.viz_frame = ttk.Frame(self.tab_visualization)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

    # ------------------------------------------------------------------
    # About tab
    # ------------------------------------------------------------------
    def _setup_about_tab(self):
        about_text = (
            "SMART AGRICULTURE DECISION SUPPORT SYSTEM\n"
            "==========================================\n\n"
            "Course:      Artificial Intelligence (BSE-6)\n"
            "Institution: Bahria University, Islamabad Campus\n"
            "Instructor:  Engr. Saad Mazhar Khan\n"
            "Paper Type:  OEL [CLO-2]\n\n"
            "System Architecture\n"
            "-------------------\n"
            "  Data Layer      → Preprocessing & Feature Engineering\n"
            "  Model Layer     → Decision Tree | KMeans | Linear Regression\n"
            "  Presentation    → Tkinter GUI with embedded Matplotlib\n\n"
            "Components\n"
            "----------\n"
            "1. Decision Tree Classifier\n"
            "   Purpose : Crop recommendation from soil/climate features\n"
            "   Metrics : Accuracy, Precision, Recall, F1-Score\n"
            "   Output  : Recommended crop + confidence probability\n\n"
            "2. KNN Clustering (K-Means)\n"
            "   Purpose : Soil profile segmentation into farm zones\n"
            "   Metrics : Silhouette Score, Davies-Bouldin Index, Inertia\n"
            "   Output  : Cluster assignment + agronomic guidance\n\n"
            "3. Linear Regression\n"
            "   Purpose : Quantitative crop yield prediction\n"
            "   Metrics : RMSE, MAE, R² Score\n"
            "   Output  : Predicted yield + 95% confidence interval\n\n"
            "Usage\n"
            "-----\n"
            "1. Train models:  python train_models.py\n"
            "2. Launch GUI:    python src/gui.py\n"
            "3. Click 'Load Models', enter parameters, click 'Predict'.\n"
        )
        sb = ttk.Scrollbar(self.tab_about)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        tw = tk.Text(self.tab_about, font=MONO_FONT,
                     yscrollcommand=sb.set, bg='#fafafa', relief=tk.FLAT)
        tw.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        sb.config(command=tw.yview)
        tw.insert(tk.END, about_text)
        tw.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _load_models(self):
        self.status_var.set("Loading models…")
        self.root.update_idletasks()
        try:
            ok = self.model_factory.load_all_models()
            # Also load the fitted preprocessor (scaler + label encoders)
            try:
                self.preprocessor.load()
            except Exception:
                pass  # preprocessor not saved yet — warn during prediction

            if ok:
                messagebox.showinfo("Success", "All models loaded successfully!")
                self._display_metrics()
                self.status_var.set("Models loaded. Ready for prediction.")
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to load models.\nRun 'python train_models.py' first."
                )
                self.status_var.set("Model load failed.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading models:\n{e}")
            self.status_var.set("Error loading models.")

    def _make_prediction(self):
        """Collect inputs, validate, scale, and run all three models."""
        # --- collect & validate ---
        input_values = []
        warnings_list = []
        for param in ['Nitrogen', 'Phosphorus', 'Potassium',
                      'Temperature', 'Humidity', 'pH', 'Rainfall']:
            raw = self.input_fields[param].get().strip()
            if not raw:
                messagebox.showerror("Input Error", f"Please enter a value for {param}.")
                return
            try:
                val = float(raw)
            except ValueError:
                messagebox.showerror("Input Error",
                                     f"'{raw}' is not a valid number for {param}.")
                return
            ok, warn_msg = validate_input(param, val)
            if not ok:
                warnings_list.append(warn_msg)
            input_values.append(val)

        if warnings_list:
            proceed = messagebox.askyesno(
                "Out-of-Range Warning",
                "Some inputs are outside expected ranges:\n\n"
                + "\n".join(warnings_list)
                + "\n\nPredictions may be unreliable. Continue anyway?"
            )
            if not proceed:
                return

        # --- scale input using the fitted scaler ---
        X_raw = np.array(input_values).reshape(1, -1)
        try:
            X_input = self.preprocessor.scale_input(X_raw)
        except Exception:
            messagebox.showwarning(
                "Scaler Not Fitted",
                "The input scaler is not fitted. Run 'python train_models.py' "
                "and restart the GUI to ensure correct scaling.\n\n"
                "Proceeding with unscaled inputs (results may be inaccurate)."
            )
            X_input = X_raw

        # --- run models ---
        self.output_text.delete(1.0, tk.END)
        sep = "=" * 68
        out = [sep,
               "  INTEGRATED AGRICULTURAL DECISION SUPPORT OUTPUT",
               sep, ""]

        # 1. Decision Tree — returns encoded int; decode to crop name
        crop_enc_pred = None
        crop_name = None
        try:
            crop_enc_pred = factory_dt = self.model_factory.dt_classifier.predict(X_input)[0]
            crop_proba = self.model_factory.dt_classifier.predict_proba(X_input)[0]
            confidence = max(crop_proba)
            # Decode encoded label → human-readable crop name
            le = self.preprocessor.label_encoders.get('label')
            crop_name = (
                le.inverse_transform([int(crop_enc_pred)])[0]
                if le else str(crop_enc_pred)
            )
            out += [
                "1. CROP RECOMMENDATION  (Decision Tree Classifier)",
                f"   Recommended Crop : {crop_name}",
                f"   Confidence       : {confidence:.2%}",
                "",
            ]
        except Exception as e:
            out += [f"1. CROP RECOMMENDATION  — Error: {e}", ""]

        # 2. KMeans Clustering
        try:
            cluster_id = int(self.model_factory.kmeans_clusterer.predict(X_input)[0])
            guidance = get_cluster_guidance(cluster_id)
            out += [
                "2. SOIL ZONE CLASSIFICATION  (KNN Clustering / K-Means)",
                f"   Assigned Cluster : {cluster_id}  ({guidance['label']})",
                f"   Guidance         : {guidance['guidance']}",
                "",
            ]
        except Exception as e:
            out += [f"2. SOIL ZONE CLASSIFICATION  — Error: {e}", ""]

        # 3. Linear Regression — uses soil features + encoded crop label
        try:
            if crop_enc_pred is not None:
                X_reg = np.append(X_input[0], int(crop_enc_pred)).reshape(1, -1)
            else:
                X_reg = X_input  # fallback

            y_pred, lo, hi = self.model_factory.lr_regressor.predict_with_ci(X_reg)
            out += [
                "3. CROP YIELD PREDICTION  (Linear Regression)",
                f"   Predicted Yield  : {y_pred:.2f} kg/hectare",
                f"   95% CI           : [{lo:.2f}  –  {hi:.2f}] kg/hectare",
                "",
            ]
        except Exception as e:
            out += [f"3. CROP YIELD PREDICTION  — Error: {e}", ""]

        out.append(sep)
        self.output_text.insert(tk.END, "\n".join(out))
        self.status_var.set("Prediction complete.")

    def _clear_inputs(self):
        for field in self.input_fields.values():
            field.delete(0, tk.END)
        self.output_text.delete(1.0, tk.END)
        self.status_var.set("Inputs cleared.")

    def _fill_example(self):
        """Populate fields with a representative example."""
        example = {
            'Nitrogen': '80', 'Phosphorus': '50', 'Potassium': '100',
            'Temperature': '25', 'Humidity': '70', 'pH': '6.5', 'Rainfall': '100',
        }
        for param, val in example.items():
            self.input_fields[param].delete(0, tk.END)
            self.input_fields[param].insert(0, val)
        self.status_var.set("Example values loaded.")

    def _display_metrics(self):
        self.metrics_text.delete(1.0, tk.END)
        sep80 = "=" * 80
        lines = [sep80, "  MODEL PERFORMANCE METRICS", sep80, ""]

        all_metrics = self.model_factory.get_all_metrics()

        # Decision Tree
        lines += ["1. DECISION TREE CLASSIFIER", "-" * 40]
        if all_metrics['decision_tree']:
            for k, v in all_metrics['decision_tree'].items():
                lines.append(f"   {k:<20}: {v:.4f}")
        else:
            lines.append("   No metrics available — train models first.")
        lines.append("")

        # KMeans
        lines += ["2. KNN CLUSTERING (K-Means)", "-" * 40]
        if all_metrics['kmeans']:
            for k, v in all_metrics['kmeans'].items():
                if isinstance(v, float):
                    lines.append(f"   {k:<20}: {v:.4f}")
                else:
                    lines.append(f"   {k:<20}: {v}")
        else:
            lines.append("   No metrics available — train models first.")
        lines.append("")

        # Linear Regression
        lines += ["3. LINEAR REGRESSION", "-" * 40]
        if all_metrics['linear_regression']:
            for k, v in all_metrics['linear_regression'].items():
                lines.append(f"   {k:<20}: {v:.4f}")
        else:
            lines.append("   No metrics available — train models first.")

        self.metrics_text.insert(tk.END, "\n".join(lines))

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    def _plot_feature_importance(self):
        try:
            fi = self.model_factory.dt_classifier.feature_importance
            if fi is None:
                messagebox.showwarning("Warning", "Decision Tree not trained yet.")
                return

            features = self.model_factory.dt_classifier.feature_names
            importance = fi

            # Sort descending
            sorted_pairs = sorted(zip(features, importance),
                                  key=lambda x: x[1], reverse=True)
            feats, imps = zip(*sorted_pairs)

            fig = Figure(figsize=(9, 5), tight_layout=True)
            ax = fig.add_subplot(111)
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feats)))
            bars = ax.barh(feats, imps, color=colors)
            ax.set_xlabel('Importance Score', fontsize=11)
            ax.set_title('Decision Tree — Feature Importance', fontsize=13, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            for bar, imp in zip(bars, imps):
                ax.text(imp + 0.002, bar.get_y() + bar.get_height() / 2,
                        f'{imp:.3f}', va='center', fontsize=9)
            self._display_figure(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting feature importance:\n{e}")

    def _plot_clusters(self):
        """Scatter plot of training data coloured by cluster assignment."""
        try:
            km = self.model_factory.kmeans_clusterer
            if km.cluster_centers is None:
                messagebox.showwarning("Warning", "KMeans model not trained yet.")
                return

            X_train = km.X_train_
            labels = km.labels_
            centers = km.cluster_centers

            if X_train is None or labels is None:
                messagebox.showwarning(
                    "Warning",
                    "Training data not available for scatter plot.\n"
                    "Retrain models to enable this visualization."
                )
                return

            fig = Figure(figsize=(9, 5), tight_layout=True)
            ax = fig.add_subplot(111)

            cmap = plt.cm.get_cmap('Set1', km.n_clusters)
            for c in range(km.n_clusters):
                mask = labels == c
                ax.scatter(X_train[mask, 0], X_train[mask, 1],
                           c=[cmap(c)], alpha=0.4, s=15, label=f'Cluster {c}')

            ax.scatter(centers[:, 0], centers[:, 1],
                       c='black', marker='X', s=250, zorder=5, label='Centroids')
            ax.set_xlabel('Feature 1 (scaled)', fontsize=11)
            ax.set_ylabel('Feature 2 (scaled)', fontsize=11)
            ax.set_title('KNN Clustering — Soil Zone Scatter (PC1 vs PC2)',
                         fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            self._display_figure(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting clusters:\n{e}")

    def _plot_residuals(self):
        """Residual plot for Linear Regression."""
        try:
            residuals = self.model_factory.lr_regressor.residuals
            predictions = self.model_factory.lr_regressor.predictions
            if residuals is None:
                messagebox.showwarning("Warning", "Linear Regression not trained yet.")
                return

            fig = Figure(figsize=(9, 5), tight_layout=True)
            ax = fig.add_subplot(111)
            ax.scatter(predictions, residuals, alpha=0.5, color='steelblue', s=20)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
            ax.set_xlabel('Predicted Yield (kg/hectare)', fontsize=11)
            ax.set_ylabel('Residual', fontsize=11)
            ax.set_title('Linear Regression — Residual Plot', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Annotate residual std
            std = np.std(residuals)
            ax.text(0.02, 0.95, f'Residual Std: {std:.2f}',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            self._display_figure(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting residuals:\n{e}")

    def _plot_actual_vs_predicted(self):
        """Actual vs Predicted scatter for Linear Regression."""
        try:
            lr = self.model_factory.lr_regressor
            if lr.predictions is None or lr.y_test_ is None:
                messagebox.showwarning("Warning", "Linear Regression not trained yet.")
                return

            y_actual = lr.y_test_
            y_pred = lr.predictions

            fig = Figure(figsize=(9, 5), tight_layout=True)
            ax = fig.add_subplot(111)
            ax.scatter(y_actual, y_pred, alpha=0.5, color='teal', s=20)
            mn = min(y_actual.min(), y_pred.min())
            mx = max(y_actual.max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
            ax.set_xlabel('Actual Yield (kg/hectare)', fontsize=11)
            ax.set_ylabel('Predicted Yield (kg/hectare)', fontsize=11)
            ax.set_title('Linear Regression — Actual vs Predicted',
                         fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            r2 = lr.metrics.get('r2_score', 0)
            ax.text(0.02, 0.95, f'R² = {r2:.4f}',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
            self._display_figure(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting actual vs predicted:\n{e}")

    def _display_figure(self, fig):
        """Embed a matplotlib Figure inside the visualization tab."""
        # Destroy previous canvas
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        self._canvas_widget = None

        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        self._canvas_widget = canvas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_gui():
    root = tk.Tk()
    app = AgricultureGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
