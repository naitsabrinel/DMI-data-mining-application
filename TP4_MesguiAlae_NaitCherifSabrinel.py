import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import confusion_matrix

NULL_VALUES = [
    "", " ", "  ", "?", "-", "_",
    "NA", "N/A", "na", "n/a",
    "null", "NULL", "None", "none", "nan"
]

#VALIDATION FUNCTIONS
def normalize_missing_values(df):
    return df.replace(NULL_VALUES, np.nan)

def validate_dataset_loaded(df):
    if df is None:
        raise ValueError("❌ No dataset loaded.")

def validate_not_empty(df):
    if df.empty:
        raise ValueError("❌ Dataset is empty.")

def validate_missing_values(df):
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        raise ValueError(
            f"❌ Dataset contains {total_missing} missing values.\n"
            "Please apply missing value preprocessing."
        )

def validate_no_categorical(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        raise ValueError(
            f"❌ Categorical columns detected: {list(cat_cols)}.\n"
            "Please apply encoding first."
        )

def validate_classification_target(y):
    if y.isnull().sum() > 0:
        raise ValueError("❌ Target column contains missing values.")

    if y.nunique() < 2:
        raise ValueError("❌ Target must contain at least 2 classes.")

    # Continuous target detection
    if y.dtype != 'object' and y.nunique() > 20:
        raise ValueError(
            "❌ Selected target is continuous.\n"
            "Classification requires a discrete target."
        )



def load_predefined(name):
    if name.lower() == "iris":
        from sklearn.datasets import load_iris
        iris = load_iris()
        return pd.DataFrame(iris.data, columns=iris.feature_names)
    elif name.lower() == "wine":
        from sklearn.datasets import load_wine
        wine = load_wine()
        return pd.DataFrame(wine.data, columns=wine.feature_names)
    else:
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=200, centers=3, random_state=0)
        return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])


class DMIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("My DMI - An Advanced Data Mining Interface")
        self.geometry("1000x700")
        self.original_target = {}

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Colors
        self.bg_color = "#f5f5f5"
        self.primary_color = "#2c3e50"
        self.secondary_color = "#3498db"
        self.accent_color = "#e74c3c"
        
        self.configure(bg=self.bg_color)
        
        self.data = None
        self.data_name = tk.StringVar(value="No dataset loaded")
        self.clustering_history = []

        # ----------------
        # Top Frame: Title + Dataset Name
        # ----------------
        top_frame = tk.Frame(self, bg=self.primary_color)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=0)
        
        title_label = tk.Label(top_frame, text="My DMI", font=("Helvetica", 28, "bold"), 
                              fg="white", bg=self.primary_color)
        title_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        dataset_status = tk.Label(top_frame, textvariable=self.data_name, 
                                 font=("Helvetica", 12), fg="#ecf0f1", bg=self.primary_color)
        dataset_status.pack(side=tk.RIGHT, padx=20)
        
        # ----------------
        # Notebook (tabs)
        # ----------------
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure notebook style
        self.style.configure('TNotebook.Tab', font=('Helvetica', 12, 'bold'), 
                           padding=[15, 8], background="#ecf0f1")
        self.style.map('TNotebook.Tab', background=[('selected', self.secondary_color)])
        
        # Tabs
        self.welcome_tab = ttk.Frame(self.notebook)
        self.preview_tab = ttk.Frame(self.notebook)
        self.preprocessing_tab = ttk.Frame(self.notebook)
        self.clustering_tab = ttk.Frame(self.notebook)
        self.classification_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.welcome_tab, text="🏠 Welcome")
        self.notebook.add(self.preview_tab, text="📊 Preview")
        self.notebook.add(self.preprocessing_tab, text="⚙️ Preprocessing")
        self.notebook.add(self.clustering_tab, text="🔍 Clustering")
        self.notebook.add(self.classification_tab, text="🧠 Classification")
        self.notebook.add(self.results_tab, text="📈 Results")
        
        # Build tabs
        self.build_welcome_tab()
        self.build_preview_tab()
        self.build_preprocessing_tab()
        self.build_clustering_tab()
        self.build_classification_tab()
        self.build_results_tab()
             
    def load_csv_dialog(self):
            path = filedialog.askopenfilename(
                filetypes=[("CSV Files", "*.csv"),
                        ("Excel Files", "*.xlsx"),
                        ("All files", "*.*")]
            )

            if not path:
                return
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                    df = normalize_missing_values(df)
                    self.data = df.copy()
                elif path.endswith('.xlsx'):
                    df = pd.read_excel(path)
                    df = normalize_missing_values(df)
                    self.data = df.copy()
                else:
                    df = pd.read_csv(path, sep=None, engine='python')
                    df = normalize_missing_values(df)
                    self.data = df.copy()

                self.data_name.set(path.split("/")[-1])
                self.show_preview()

                messagebox.showinfo(
                    "Success",
                    f"Dataset loaded successfully!\nShape: {self.data.shape}"
                )

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")
    

    def load_predefined_dataset(self):
            choice = self.predefined_var.get()
            df = load_predefined(choice)
            df = normalize_missing_values(df)
            self.data = df.copy()

            self.data_name.set(f"Predefined: {choice}")
            self.show_preview()
            messagebox.showinfo("Success", f"{choice} dataset loaded successfully!\nShape: {self.data.shape}")
    # ------------------------
    # Welcome Tab
    def build_welcome_tab(self):
        welcome_bg = tk.Frame(self.welcome_tab, bg=self.bg_color)
        welcome_bg.pack(fill=tk.BOTH, expand=True)
        
        lbl_title = tk.Label(welcome_bg, text="Welcome to My DMI", 
                            font=("Helvetica", 32, "bold"), fg=self.primary_color, bg=self.bg_color)
        lbl_title.pack(pady=40)
        
        lbl_desc = tk.Label(welcome_bg, text=(
            "My DMI is your interactive data mining and clustering tool.\n\n"
            "Load your dataset, preprocess it with multiple methods, perform clustering,\n"
            "and visualize results with comprehensive metrics and analysis."
        ), font=("Helvetica", 16), justify="center", bg=self.bg_color, fg="#333333")
        lbl_desc.pack(pady=20)
        
        # Get Started Button
        get_started_btn = tk.Button(welcome_bg, text="Get Started →", 
                                   font=("Helvetica", 14, "bold"),
                                   bg=self.secondary_color, fg="white",
                                   command=lambda: self.notebook.select(self.preview_tab))
        get_started_btn.pack(pady=30)

    # ------------------------
    # Preview Tab
    def build_preview_tab(self):
        # Main container
        main_container = tk.Frame(self.preview_tab)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel
        left_panel = tk.Frame(main_container, bg="#ecf0f1", relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Load buttons
        btn_frame = tk.Frame(left_panel, bg="#ecf0f1")
        btn_frame.pack(fill='x', pady=10, padx=10)
        
        tk.Button(btn_frame, text="📂 Load CSV file...", font=("Helvetica", 12),
                 bg=self.secondary_color, fg="white", command=self.load_csv_dialog).pack(fill='x', pady=5)
        
        tk.Label(btn_frame, text="Or choose predefined dataset:", 
                font=("Helvetica", 11), bg="#ecf0f1").pack(anchor="w", pady=(10, 2))
        
        self.predefined_var = tk.StringVar(value="Iris")
        predefined_options = ["Iris", "Wine", "Blobs"]
        self.predefined_combo = ttk.Combobox(btn_frame, values=predefined_options, 
                                            state="readonly", textvariable=self.predefined_var, 
                                            font=("Helvetica", 11))
        self.predefined_combo.pack(fill='x', pady=2)
        
        tk.Button(btn_frame, text="📥 Load Dataset", font=("Helvetica", 12),
                 bg="#27ae60", fg="white", command=self.load_predefined_dataset).pack(fill='x', pady=5)
        
        # Right panel
        right_panel = tk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Dataset label
        self.dataset_label_preview = tk.Label(right_panel, text="Dataset: None", 
                                             font=("Helvetica", 14, "bold"), fg=self.primary_color)
        self.dataset_label_preview.pack(anchor="w", padx=10, pady=(5, 10))
        
        # Notebook for data views
        preview_notebook = ttk.Notebook(right_panel)
        preview_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Data Head
        head_frame = ttk.Frame(preview_notebook)
        preview_notebook.add(head_frame, text="Data Head")
        
        head_scroll = tk.Scrollbar(head_frame)
        head_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.head_tree = ttk.Treeview(head_frame, show="headings", height=10, 
                                     yscrollcommand=head_scroll.set)
        self.head_tree.pack(fill=tk.BOTH, expand=True)
        head_scroll.config(command=self.head_tree.yview)
        self.head_tree.bind("<Key>", lambda e: "break")
        
        # Tab 2: Info & Stats
        info_stats_frame = ttk.Frame(preview_notebook)
        preview_notebook.add(info_stats_frame, text="Info & Stats")
        
        # Info section
        info_frame = ttk.LabelFrame(info_stats_frame, text="Dataset Info", padding=10)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        info_scroll = tk.Scrollbar(info_frame)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.info_tree = ttk.Treeview(info_frame, columns=("Attribute", "Type", "Missing"), 
                                     show="headings", height=15, yscrollcommand=info_scroll.set)
        for col in ["Attribute", "Type", "Missing"]:
            self.info_tree.heading(col, text=col, anchor='center')
            self.info_tree.column(col, anchor='center', width=120)
        self.info_tree.pack(fill=tk.BOTH, expand=True)
        info_scroll.config(command=self.info_tree.yview)
        self.info_tree.bind("<Key>", lambda e: "break")
        
        # Stats section
        stat_frame = ttk.LabelFrame(info_stats_frame, text="Statistics", padding=10)
        stat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        stat_scroll = tk.Scrollbar(stat_frame)
        stat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stat_tree = ttk.Treeview(stat_frame, 
                                     columns=("Attribute", "Min", "Q1", "Median", "Q3", "Max", "Mean", "Mode"), 
                                     show="headings", height=15, yscrollcommand=stat_scroll.set)
        for col in ["Attribute", "Min", "Q1", "Median", "Q3", "Max", "Mean", "Mode"]:
            self.stat_tree.heading(col, text=col, anchor='center')
            self.stat_tree.column(col, anchor='center', width=90)
        self.stat_tree.pack(fill=tk.BOTH, expand=True)
        stat_scroll.config(command=self.stat_tree.yview)
        self.stat_tree.bind("<Key>", lambda e: "break")

    # ------------------------
    # Preprocessing Tab
    def build_preprocessing_tab(self):
        # Create a container with scrollbars for preprocessing tab
        preproc_container = tk.Frame(self.preprocessing_tab)
        preproc_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for preprocessing tab
        preproc_canvas = tk.Canvas(preproc_container, bg=self.bg_color)
        preproc_v_scroll = tk.Scrollbar(preproc_container, orient="vertical", command=preproc_canvas.yview)
        preproc_h_scroll = tk.Scrollbar(preproc_container, orient="horizontal", command=preproc_canvas.xview)
        
        preproc_scrollable = tk.Frame(preproc_canvas, bg=self.bg_color)
        preproc_scrollable.bind(
            "<Configure>",
            lambda e: preproc_canvas.configure(scrollregion=preproc_canvas.bbox("all"))
        )
        
        preproc_canvas_window = preproc_canvas.create_window((0, 0), window=preproc_scrollable, anchor="nw")
        preproc_canvas.configure(yscrollcommand=preproc_v_scroll.set, xscrollcommand=preproc_h_scroll.set)
        
        # Pack scrollbars and canvas
        preproc_v_scroll.pack(side="right", fill="y")
        preproc_h_scroll.pack(side="bottom", fill="x")
        preproc_canvas.pack(side="left", fill="both", expand=True)
        
        # Bind mouse wheel for this canvas
        preproc_canvas.bind_all("<MouseWheel>", lambda e: preproc_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        preproc_canvas.bind("<Configure>", lambda e: preproc_canvas.itemconfig(preproc_canvas_window, width=e.width))
        
        main_frame = tk.Frame(preproc_scrollable)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls (now smaller and more compact)
        left_panel = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Dataset info (smaller font)
        self.dataset_label_preprocessing = tk.Label(left_panel, text="Dataset: None", 
                                                   font=("Helvetica", 12, "bold"), 
                                                   fg=self.primary_color, bg="#ecf0f1")
        self.dataset_label_preprocessing.pack(anchor="w", padx=10, pady=(10, 5))
        
        # Create a canvas for the options with scrollbar
        options_canvas = tk.Canvas(left_panel, bg="#ecf0f1", highlightthickness=0)
        options_scrollbar = tk.Scrollbar(left_panel, orient="vertical", command=options_canvas.yview)
        
        options_frame = tk.Frame(options_canvas, bg="#ecf0f1")
        options_frame.bind(
            "<Configure>",
            lambda e: options_canvas.configure(scrollregion=options_canvas.bbox("all"))
        )
        
        options_canvas_window = options_canvas.create_window((0, 0), window=options_frame, anchor="nw")
        options_canvas.configure(yscrollcommand=options_scrollbar.set)
        
        # Pack the canvas and scrollbar
        options_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        options_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse wheel for options canvas
        options_canvas.bind_all("<MouseWheel>", lambda e: options_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        options_canvas.bind("<Configure>", lambda e: options_canvas.itemconfig(options_canvas_window, width=e.width))
        
        # Preprocessing options with smaller fonts and tighter spacing
        tk.Label(options_frame, text="Preprocessing Options:", 
                font=("Helvetica", 11, "bold"), bg="#ecf0f1").pack(anchor="w", pady=(5, 10), padx=10)
        
        # 1. Convert to numeric
        convert_frame = ttk.LabelFrame(options_frame, text="1. Convert Categorical Data", padding=8)
        convert_frame.pack(fill='x', pady=5, padx=10)
        
        self.convert_var = tk.StringVar(value="One-Hot Encoding")
        
        tk.Radiobutton(convert_frame, text="One-Hot Encoding", variable=self.convert_var, 
                      value="One-Hot Encoding", bg="#ecf0f1", font=("Helvetica", 10)).pack(anchor="w", pady=2)
        tk.Radiobutton(convert_frame, text="Label Encoding", variable=self.convert_var, 
                      value="Label Encoding", bg="#ecf0f1", font=("Helvetica", 10)).pack(anchor="w", pady=2)
        
        tk.Button(convert_frame, text="Apply Conversion", font=("Helvetica", 10),
                 bg=self.secondary_color, fg="white", command=self.convert_to_numeric).pack(fill='x', pady=5)
        
        # 2. Handle missing values
        missing_frame = ttk.LabelFrame(options_frame, text="2. Handle Missing Values", padding=8)
        missing_frame.pack(fill='x', pady=5, padx=10)
        
        self.missing_var = tk.StringVar(value="Mean")
        
        tk.Radiobutton(missing_frame, text="Mean Imputation", variable=self.missing_var, 
                      value="Mean", bg="#ecf0f1", font=("Helvetica", 10)).pack(anchor="w", pady=2)
        tk.Radiobutton(missing_frame, text="Median Imputation", variable=self.missing_var, 
                      value="Median", bg="#ecf0f1", font=("Helvetica", 10)).pack(anchor="w", pady=2)
        tk.Radiobutton(missing_frame, text="Mode Imputation", variable=self.missing_var, 
                      value="Mode", bg="#ecf0f1", font=("Helvetica", 10)).pack(anchor="w", pady=2)
        tk.Radiobutton(missing_frame, text="Drop (>60% missing)", 
                      variable=self.missing_var, value="Drop", bg="#ecf0f1", font=("Helvetica", 10)).pack(anchor="w", pady=2)
        
        tk.Button(missing_frame, text="Apply Missing Values Handling", font=("Helvetica", 10),
                 bg=self.secondary_color, fg="white", command=self.clean_missing_values).pack(fill='x', pady=5)
        
        # 3. Handle duplicates
        dup_frame = ttk.LabelFrame(options_frame, text="3. Handle Duplicates", padding=8)
        dup_frame.pack(fill='x', pady=5, padx=10)
        
        tk.Button(dup_frame, text="Remove Duplicate Rows", font=("Helvetica", 10),
                 bg=self.secondary_color, fg="white", command=self.handle_duplicates).pack(fill='x', pady=5)
        
        # 4. Feature scaling
        scaling_frame = ttk.LabelFrame(options_frame, text="4. Feature Scaling", padding=8)
        scaling_frame.pack(fill='x', pady=5, padx=10)
        
        self.scaling_var = tk.StringVar(value="Min-Max")
        
        tk.Radiobutton(scaling_frame, text="Min-Max Scaling", variable=self.scaling_var, 
                      value="Min-Max", bg="#ecf0f1", font=("Helvetica", 10)).pack(anchor="w", pady=2)
        tk.Radiobutton(scaling_frame, text="Robust Scaling", variable=self.scaling_var, 
                      value="Robust", bg="#ecf0f1", font=("Helvetica", 10)).pack(anchor="w", pady=2)
        tk.Radiobutton(scaling_frame, text="Z-Score (Standard)", variable=self.scaling_var, 
                      value="Z-Score", bg="#ecf0f1", font=("Helvetica", 10)).pack(anchor="w", pady=2)
        
        tk.Button(scaling_frame, text="Apply Feature Scaling", font=("Helvetica", 10),
                 bg=self.secondary_color, fg="white", command=self.feature_scaling).pack(fill='x', pady=5)
        
        # Update the options frame width
        options_canvas.update_idletasks()
        options_canvas.config(width=320)  # Fixed width for the options panel
        
        # Right panel - Results display
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        results_frame = ttk.LabelFrame(right_panel, text="📋 Preprocessing Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text widget with scrollbars
        results_container = tk.Frame(results_frame)
        results_container.pack(fill=tk.BOTH, expand=True)
        
        results_scroll_y = tk.Scrollbar(results_container)
        results_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        results_scroll_x = tk.Scrollbar(results_container, orient=tk.HORIZONTAL)
        results_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.preproc_display_text = tk.Text(
            results_container,
            height=20,
            wrap="word",
            bg="#f9f9f9",
            fg="#2c3e50",
            font=("Segoe UI", 10),
            padx=12,
            pady=12,
            yscrollcommand=results_scroll_y.set,
            xscrollcommand=results_scroll_x.set
        )
        self.preproc_display_text.config(state="disabled")
        self.preproc_display_text.tag_config(
            "title",
            font=("Segoe UI", 11, "bold"),
            foreground="#1f4fd8"
        )

        self.preproc_display_text.tag_config(
            "section",
            font=("Segoe UI", 10, "bold"),
            foreground="#2c3e50"
        )

        self.preproc_display_text.tag_config(
            "success",
            foreground="#2e7d32"
        )

        self.preproc_display_text.tag_config(
            "warning",
            foreground="#e67e22"
        )

        self.preproc_display_text.tag_config(
            "separator",
            foreground="#b0b0b0"
        )

        self.preproc_display_text.pack(fill=tk.BOTH, expand=True)
        results_scroll_y.config(command=self.preproc_display_text.yview)
        results_scroll_x.config(command=self.preproc_display_text.xview)
        
        clear_btn = tk.Button(
            right_panel,
            text="🗑 Clear History",
            bg="#d9534f",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            command=self.clear_preprocessing_history
        )
        clear_btn.pack(pady=6)

    # ------------------------
    # Preprocessing functions
    def convert_to_numeric(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No dataset loaded")
            return
        
        method = self.convert_var.get()
        result_text = f"Applying {method}...\n"
        
        if method == "One-Hot Encoding":
            # Get categorical columns
            cat_cols = self.data.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                # Apply one-hot encoding
                dummies = pd.get_dummies(self.data[cat_cols], drop_first=True)
                self.data = pd.concat([self.data.drop(cat_cols, axis=1), dummies], axis=1)
                result_text += f"Applied One-Hot Encoding to columns: {list(cat_cols)}\n"
                result_text += f"Created {len(dummies.columns)} new binary columns\n"
            else:
                result_text += "No categorical columns found for One-Hot Encoding\n"
        
        elif method == "Label Encoding":
            from sklearn.preprocessing import LabelEncoder
            cat_cols = self.data.select_dtypes(include=['object']).columns
            le = LabelEncoder()
            converted_cols = []
            
            for col in cat_cols:
                try:
                    self.data[col] = le.fit_transform(self.data[col])
                    converted_cols.append(col)
                except:
                    pass
            
            result_text += f"Applied Label Encoding to columns: {converted_cols}\n"
        
        self.update_results_display(result_text)
        self.show_preview()

    def clean_missing_values(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No dataset loaded")
            return
        
        method = self.missing_var.get()
        result_text = f"Handling missing values using {method}...\n"
        
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            total = len(self.data)
            
            if missing_count > 0:
                # Check if > 60% missing
                if missing_count / total > 0.6 and method == "Drop":
                    self.data.drop(columns=[col], inplace=True)
                    result_text += f"Dropped column '{col}' ({missing_count} missing, >60%)\n"
                    continue
                
                if method == "Mean":
                    if self.data[col].dtype in ['float64', 'int64']:
                        self.data[col].fillna(self.data[col].mean(), inplace=True)
                        result_text += f"Column '{col}': filled {missing_count} missing with mean\n"
                
                elif method == "Median":
                    if self.data[col].dtype in ['float64', 'int64']:
                        self.data[col].fillna(self.data[col].median(), inplace=True)
                        result_text += f"Column '{col}': filled {missing_count} missing with median\n"
                
                elif method == "Mode":
                    if self.data[col].dtype in ['float64', 'int64']:
                        self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                        result_text += f"Column '{col}': filled {missing_count} missing with mode\n"
        
        if "No missing values" in result_text:
            result_text = "No missing values found in the dataset.\n"
        
        self.update_results_display(result_text)
        self.show_preview()

    def handle_duplicates(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No dataset loaded")
            return
        
        dup_count = self.data.duplicated().sum()
        
        if dup_count > 0:
            ans = messagebox.askyesno("Duplicate Rows", 
                                     f"Found {dup_count} duplicate rows. Do you want to remove them?")
            if ans:
                self.data.drop_duplicates(inplace=True)
                result_text = f"Removed {dup_count} duplicate rows.\n"
            else:
                result_text = f"Found {dup_count} duplicate rows (not removed).\n"
        else:
            result_text = "No duplicate rows found.\n"
        
        self.update_results_display(result_text)
        self.show_preview()

    def feature_scaling(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No dataset loaded")
            return
        
        method = self.scaling_var.get()
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            messagebox.showinfo("Scaling", "No numeric columns to scale.")
            return
        
        if method == "Min-Max":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
            result_text = f"Applied Min-Max scaling to columns: {list(numeric_cols)}\n"
        
        elif method == "Robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
            result_text = f"Applied Robust scaling to columns: {list(numeric_cols)}\n"
        
        elif method == "Z-Score":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
            result_text = f"Applied Z-Score (Standard) scaling to columns: {list(numeric_cols)}\n"
        
        self.update_results_display(result_text)
        self.show_preview()

    def update_results_display(self, text):
        self.preproc_display_text.config(state="normal")
        
        # Insert a separator if there's already content
        current_content = self.preproc_display_text.get("1.0", tk.END).strip()
        if current_content:
            self.preproc_display_text.insert(tk.END, "\n" + "=" * 60 + "\n\n", "separator")
        
        # Insert the actual preprocessing result
        self.preproc_display_text.insert(tk.END, "📊 Preprocessing Step\n", "title")
        self.preproc_display_text.insert(tk.END, "-" * 60 + "\n", "separator")
        self.preproc_display_text.insert(tk.END, text + "\n", "section")
        
        self.preproc_display_text.config(state="disabled")
        self.preproc_display_text.see(tk.END)
            
    def clear_preprocessing_history(self):
        self.preproc_display_text.config(state="normal")
        self.preproc_display_text.delete("1.0", tk.END)
        self.preproc_display_text.config(state="disabled")

        messagebox.showinfo(
            "Preprocessing",
            "Preprocessing history cleared."
        )


    # ------------------------
    # Clustering Tab
    def build_clustering_tab(self):
        main_frame = tk.Frame(self.clustering_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls with scrollbar
        left_panel_container = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.RAISED, bd=2)
        left_panel_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Create canvas for left panel with scrollbar
        left_panel_canvas = tk.Canvas(left_panel_container, bg="#ecf0f1", highlightthickness=0)
        left_panel_scrollbar = tk.Scrollbar(left_panel_container, orient="vertical", command=left_panel_canvas.yview)
        
        left_panel = tk.Frame(left_panel_canvas, bg="#ecf0f1")
        left_panel.bind(
            "<Configure>",
            lambda e: left_panel_canvas.configure(scrollregion=left_panel_canvas.bbox("all"))
        )
        
        left_panel_canvas_window = left_panel_canvas.create_window((0, 0), window=left_panel, anchor="nw")
        left_panel_canvas.configure(yscrollcommand=left_panel_scrollbar.set)
        
        # Pack the canvas and scrollbar
        left_panel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_panel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse wheel for left panel canvas
        left_panel_canvas.bind_all("<MouseWheel>", lambda e: left_panel_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        left_panel_canvas.bind("<Configure>", lambda e: left_panel_canvas.itemconfig(left_panel_canvas_window, width=e.width))
        
        # Dataset info
        self.dataset_label_clustering = tk.Label(left_panel, text="Dataset: None", 
                                                font=("Helvetica", 12, "bold"), 
                                                fg=self.primary_color, bg="#ecf0f1")
        self.dataset_label_clustering.pack(anchor="w", padx=10, pady=(10, 5))
        
        # Algorithm selection
        tk.Label(left_panel, text="Algorithm:", font=("Helvetica", 11, "bold"), 
                bg="#ecf0f1").pack(anchor="w", padx=10, pady=(10, 2))
        
        self.clust_algo_var = tk.StringVar(value="KMeans")
        algo_options = ["KMeans", "KMedoids", "Agglomerative", "Diana", "DBSCAN"]
        self.algo_combo = ttk.Combobox(left_panel, values=algo_options, 
                                      state="readonly", textvariable=self.clust_algo_var, 
                                      font=("Helvetica", 10))
        self.algo_combo.pack(fill='x', padx=10, pady=2)
        self.algo_combo.bind('<<ComboboxSelected>>', self.update_parameters)
        
        # Parameters frame
        self.param_frame = ttk.LabelFrame(left_panel, text="Parameters", padding=10)
        self.param_frame.pack(fill='x', padx=10, pady=10)
        
        # Initial parameters
        tk.Label(self.param_frame, text="n_clusters:", font=("Helvetica", 10)).pack(anchor="w")
        self.n_clusters_var = tk.IntVar(value=3)
        tk.Entry(self.param_frame, textvariable=self.n_clusters_var, 
                font=("Helvetica", 10)).pack(fill='x', pady=2)
        
        tk.Label(self.param_frame, text="eps (DBSCAN):", font=("Helvetica", 10)).pack(anchor="w")
        self.eps_var = tk.DoubleVar(value=0.5)
        self.eps_entry = tk.Entry(self.param_frame, textvariable=self.eps_var, 
                                 font=("Helvetica", 10))
        self.eps_entry.pack(fill='x', pady=2)
        
        tk.Label(self.param_frame, text="min_samples (DBSCAN):", font=("Helvetica", 10)).pack(anchor="w")
        self.min_samples_var = tk.IntVar(value=5)
        self.min_samples_entry = tk.Entry(self.param_frame, textvariable=self.min_samples_var, 
                                         font=("Helvetica", 10))
        self.min_samples_entry.pack(fill='x', pady=2)
        
        # Dimension selection
        tk.Label(left_panel, text="Visualization:", font=("Helvetica", 11, "bold"), 
                bg="#ecf0f1").pack(anchor="w", padx=10, pady=(10, 2))
        
        self.dimension_var = tk.StringVar(value="2D")
        dim_frame = tk.Frame(left_panel, bg="#ecf0f1")
        dim_frame.pack(fill='x', padx=10)
        
        tk.Radiobutton(dim_frame, text="2D", variable=self.dimension_var, 
                      value="2D", bg="#ecf0f1", font=("Helvetica", 10)).pack(side=tk.LEFT)
        tk.Radiobutton(dim_frame, text="3D", variable=self.dimension_var, 
                      value="3D", bg="#ecf0f1", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=10)
        
        # Analysis buttons
        analysis_frame = ttk.LabelFrame(left_panel, text="Analysis Tools", padding=10)
        analysis_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(analysis_frame, text="📊 Show Elbow Curve", font=("Helvetica", 10),
                 bg="#9b59b6", fg="white", command=self.plot_elbow).pack(fill='x', pady=2)
        tk.Button(analysis_frame, text="🌳 Show Dendrogram", font=("Helvetica", 10),
                 bg="#9b59b6", fg="white", command=self.plot_dendrogram).pack(fill='x', pady=2)
        
        # Action buttons
        action_frame = tk.Frame(left_panel, bg="#ecf0f1")
        action_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(action_frame, text="🚀 Run Clustering", font=("Helvetica", 12, "bold"),
                 bg=self.secondary_color, fg="white", command=self.run_clustering).pack(fill='x', pady=5)
        
        tk.Button(action_frame, text="💾 Export Labels to CSV", font=("Helvetica", 10),
                 bg="#27ae60", fg="white", command=self.export_labels).pack(fill='x', pady=2)
        
        # Set fixed width for left panel
        left_panel_canvas.update_idletasks()
        left_panel_canvas.config(width=320)
        
        # Right panel - Visualizations (NON-SCROLLABLE as requested)
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Notebook for different views
        self.cluster_notebook = ttk.Notebook(right_panel)
        self.cluster_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Clustering Plot
        self.plot_frame = ttk.Frame(self.cluster_notebook)
        self.cluster_notebook.add(self.plot_frame, text="📈 Clustering Plot")
        
        # Tab 2: Metrics
        self.metrics_frame = ttk.Frame(self.cluster_notebook)
        self.cluster_notebook.add(self.metrics_frame, text="📊 Metrics")
        self.build_metrics_table()
        
        # Tab 3: Cluster Info
        self.cluster_info_frame = ttk.Frame(self.cluster_notebook)
        self.cluster_notebook.add(self.cluster_info_frame, text="🔍 Cluster Details")
        
        # Initialize with default parameters
        self.update_parameters()

    def update_parameters(self, event=None):
        # Clear parameter frame
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        algo = self.clust_algo_var.get()
        
        tk.Label(self.param_frame, text=f"{algo} Parameters:", 
                font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        if algo in ["KMeans", "KMedoids", "Agglomerative", "Diana"]:
            tk.Label(self.param_frame, text="n_clusters:", font=("Helvetica", 10)).pack(anchor="w")
            self.n_clusters_var = tk.IntVar(value=3)
            tk.Entry(self.param_frame, textvariable=self.n_clusters_var, 
                    font=("Helvetica", 10)).pack(fill='x', pady=2)
        
        elif algo == "DBSCAN":
            tk.Label(self.param_frame, text="eps:", font=("Helvetica", 10)).pack(anchor="w")
            self.eps_var = tk.DoubleVar(value=0.5)
            tk.Entry(self.param_frame, textvariable=self.eps_var, 
                    font=("Helvetica", 10)).pack(fill='x', pady=2)
            
            tk.Label(self.param_frame, text="min_samples:", font=("Helvetica", 10)).pack(anchor="w")
            self.min_samples_var = tk.IntVar(value=5)
            tk.Entry(self.param_frame, textvariable=self.min_samples_var, 
                    font=("Helvetica", 10)).pack(fill='x', pady=2)

    def build_metrics_table(self):
        # Create Treeview for metrics
        metrics_container = tk.Frame(self.metrics_frame)
        metrics_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        metrics_scroll_y = tk.Scrollbar(metrics_container)
        metrics_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        metrics_scroll_x = tk.Scrollbar(metrics_container, orient=tk.HORIZONTAL)
        metrics_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.metrics_table = ttk.Treeview(metrics_container, 
                                         columns=("Algorithm", "Parameters", "Inertia", 
                                                 "Silhouette", "#Clusters", "Noise"),
                                         show="headings", height=15,
                                         yscrollcommand=metrics_scroll_y.set,
                                         xscrollcommand=metrics_scroll_x.set)
        
        columns = [("Algorithm", 120), ("Parameters", 200), ("Inertia", 100), 
                  ("Silhouette", 100), ("#Clusters", 80), ("Noise", 80)]
        
        for col, width in columns:
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, width=width, anchor='center')
        
        self.metrics_table.pack(fill=tk.BOTH, expand=True)
        metrics_scroll_y.config(command=self.metrics_table.yview)
        metrics_scroll_x.config(command=self.metrics_table.xview)
        
        # Clear button
        clear_btn = tk.Button(self.metrics_frame, text="Clear History", 
                             font=("Helvetica", 10), bg=self.accent_color, fg="white",
                             command=self.clear_metrics_history)
        clear_btn.pack(pady=10)

    def clear_metrics_history(self):
        for item in self.metrics_table.get_children():
            self.metrics_table.delete(item)
        self.clustering_history = []
        
    # ------------------------
    # Clustering Algorithms
    @staticmethod
    def apply_kmeans(X, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        inertie_kmeans = kmeans.inertia_
        silhouette_kmeans = silhouette_score(X, cluster_labels) if n_clusters > 1 else np.nan
        
        return cluster_labels, kmeans.cluster_centers_, inertie_kmeans, silhouette_kmeans
    
    @staticmethod
    def apply_kmedoids(X, n_clusters):
        X_array = X if isinstance(X, np.ndarray) else X.values
        initial_medoids = list(range(min(n_clusters, len(X_array))))
        
        kmedoids_instance = kmedoids(X_array.tolist(), initial_medoids)
        kmedoids_instance.process()
        
        clusters = kmedoids_instance.get_clusters()
        labels = np.zeros(len(X_array), dtype=int)
        
        for cluster_id, cluster_points in enumerate(clusters):
            for index in cluster_points:
                labels[index] = cluster_id
        
        final_medoids = kmedoids_instance.get_medoids()
        medoids = X_array[final_medoids]
        
        # Calculate inertia
        inertia = 0
        for i in range(n_clusters):
            points_cluster = X_array[labels == i]
            if len(points_cluster) > 0:
                distances = np.sum((points_cluster - medoids[i]) ** 2)
                inertia += np.sum(distances)
        
        silhouette = silhouette_score(X_array, labels) if n_clusters > 1 else np.nan
        return labels, medoids, inertia, silhouette
    
    @staticmethod
    def agnes_clustering(X, n_clusters):
        agnes_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = agnes_model.fit_predict(X)
        
        # Calculate centroids
        centroids = []
        for i in range(n_clusters):
            points_cluster = X[labels == i]
            if len(points_cluster) > 0:
                centroids.append(np.mean(points_cluster, axis=0))
        centroids = np.array(centroids) if centroids else None
        
        # Calculate inertia
        inertia = 0
        for i in range(n_clusters):
            points_cluster = X[labels == i]
            if len(points_cluster) > 0:
                centroid = np.mean(points_cluster, axis=0)
                distances = np.sum((points_cluster - centroid) ** 2)
                inertia += np.sum(distances)
        
        silhouette = silhouette_score(X, labels) if n_clusters > 1 else np.nan
        return labels, centroids, inertia, silhouette
    
    @staticmethod
    def diana_clustering(X, n_clusters):
        Z = linkage(X, method='complete')
        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
        
        # Calculate centroids
        centroids = []
        for i in range(n_clusters):
            points_cluster = X[labels == i]
            if len(points_cluster) > 0:
                centroids.append(np.mean(points_cluster, axis=0))
        centroids = np.array(centroids) if centroids else None
        
        # Calculate inertia
        inertia = 0
        unique_labels = np.unique(labels)
        for label in unique_labels:
            points_cluster = X[labels == label]
            if len(points_cluster) > 0:
                centroid = np.mean(points_cluster, axis=0)
                distances = np.sum((points_cluster - centroid) ** 2)
                inertia += np.sum(distances)
        
        silhouette = silhouette_score(X, labels) if len(unique_labels) > 1 else np.nan
        return labels, centroids, inertia, silhouette
    
    @staticmethod
    def apply_dbscan(X, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate centroids for non-noise clusters
        centroids = []
        for i in range(n_clusters):
            points_cluster = X[labels == i]
            if len(points_cluster) > 0:
                centroids.append(np.mean(points_cluster, axis=0))
        centroids = np.array(centroids) if centroids else None
        
        # Calculate inertia (excluding noise)
        inertia = 0
        for i in range(n_clusters):
            points_cluster = X[labels == i]
            if len(points_cluster) > 0:
                centroid = np.mean(points_cluster, axis=0)
                distances = np.sum((points_cluster - centroid) ** 2)
                inertia += np.sum(distances)
        
        # Calculate silhouette (excluding noise)
        mask = labels != -1
        if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
            silhouette = silhouette_score(X[mask], labels[mask])
        else:
            silhouette = np.nan
        
        return labels, centroids, inertia, silhouette, n_noise
  
    # ------------------------
    # Analysis Functions
    def plot_elbow(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No dataset loaded")
            return
        
        X = self.data.select_dtypes(include=[np.number]).values
        if X.shape[1] < 1:
            messagebox.showwarning("Warning", "Dataset must have at least one numeric column")
            return
        
        inertias = []
        K_range = range(1, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Clear plot frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Inertia', fontsize=12)
        ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.cluster_notebook.select(self.plot_frame)

    def plot_dendrogram(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No dataset loaded")
            return
        
        X = self.data.select_dtypes(include=[np.number]).values
        if X.shape[1] < 1:
            messagebox.showwarning("Warning", "Dataset must have at least one numeric column")
            return
        
        # Use a sample if dataset is large
        if len(X) > 100:
            indices = np.random.choice(len(X), 100, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        Z = linkage(X_sample, method='ward')
        
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        fig = plt.Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        dendrogram(Z, ax=ax, leaf_rotation=90., leaf_font_size=8.)
        ax.set_title('Dendrogram', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample index', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.cluster_notebook.select(self.plot_frame)

    def run_clustering(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No dataset loaded")
            return
        
        X = self.data.select_dtypes(include=[np.number]).values
        if X.shape[1] < 1:
            messagebox.showwarning("Warning", "Dataset must have at least one numeric column")
            return
        
        algo = self.clust_algo_var.get()
        
        try:
            validate_missing_values(self.data)
            validate_no_categorical(self.data)
        except ValueError as e:
            messagebox.showerror("Clustering Error", str(e))

        try:
            if algo == "KMeans":
                n_clusters = self.n_clusters_var.get()
                labels, centers, inertia, silhouette = self.apply_kmeans(X, n_clusters)
                params = f"n_clusters={n_clusters}"
                noise = 0
            
            elif algo == "KMedoids":
                n_clusters = self.n_clusters_var.get()
                labels, centers, inertia, silhouette = self.apply_kmedoids(X, n_clusters)
                params = f"n_clusters={n_clusters}"
                noise = 0
            
            elif algo == "Agglomerative":
                n_clusters = self.n_clusters_var.get()
                labels, centers, inertia, silhouette = self.agnes_clustering(X, n_clusters)
                params = f"n_clusters={n_clusters}, linkage=ward"
                noise = 0
            
            elif algo == "Diana":
                n_clusters = self.n_clusters_var.get()
                labels, centers, inertia, silhouette = self.diana_clustering(X, n_clusters)
                params = f"n_clusters={n_clusters}, linkage=complete"
                noise = 0
            
            elif algo == "DBSCAN":
                eps = self.eps_var.get()
                min_samples = self.min_samples_var.get()
                labels, centers, inertia, silhouette, noise = self.apply_dbscan(X, eps, min_samples)
                params = f"eps={eps}, min_samples={min_samples}"
            
            else:
                messagebox.showwarning("Warning", f"Unknown algorithm: {algo}")
                return
            
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Store in history
            result = {
                'algorithm': algo,
                'parameters': params,
                'inertia': inertia,
                'silhouette': silhouette,
                'n_clusters': n_clusters_found,
                'noise': noise,
                'labels': labels
            }
            self.clustering_history.append(result)
            
            # Update metrics table
            self.update_metrics_table(result)
            
            # Show plot
            self.plot_clustering_result(X, labels, centers, algo)
            
            # Show cluster details
            self.show_cluster_details(labels, n_clusters_found, noise)
            
            # Switch to plot tab
            self.cluster_notebook.select(self.plot_frame)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during clustering: {str(e)}")

    def update_metrics_table(self, result):
        # Clear existing items if this is the first result
        if len(self.clustering_history) == 1:
            for item in self.metrics_table.get_children():
                self.metrics_table.delete(item)
        
        # Format values
        inertia = f"{result['inertia']:.2f}" if not np.isnan(result['inertia']) else "N/A"
        silhouette = f"{result['silhouette']:.3f}" if not np.isnan(result['silhouette']) else "N/A"
        noise = result['noise'] if 'noise' in result else 0
        
        # Insert new row
        self.metrics_table.insert("", "end", 
                                 values=(result['algorithm'], result['parameters'], 
                                        inertia, silhouette, result['n_clusters'], noise))

    def plot_clustering_result(self, X, labels, centers, algo):
        # Clear plot frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        fig = plt.Figure(figsize=(8, 6))
        
        if self.dimension_var.get() == "3D" and X.shape[1] >= 3:
            ax = fig.add_subplot(111, projection='3d')
            xs, ys, zs = X[:, 0], X[:, 1], X[:, 2]
            
            # Create scatter plot
            scatter = ax.scatter(xs, ys, zs, c=labels, cmap='tab10', s=50, alpha=0.7)
            
            # Add centers if available
            if centers is not None and len(centers) > 0:
                if X.shape[1] >= 3:
                    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                              c='black', marker='X', s=200, label='Centroids')
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
        
        else:
            ax = fig.add_subplot(111)
            xs = X[:, 0]
            ys = X[:, 1] if X.shape[1] > 1 else X[:, 0]
            
            # Create scatter plot
            scatter = ax.scatter(xs, ys, c=labels, cmap='tab10', s=50, alpha=0.7)
            
            # Add centers if available
            if centers is not None and len(centers) > 0:
                if X.shape[1] >= 2:
                    ax.scatter(centers[:, 0], centers[:, 1], 
                              c='black', marker='X', s=200, label='Centroids')
                else:
                    ax.scatter(centers[:, 0], [0]*len(centers), 
                              c='black', marker='X', s=200, label='Centroids')
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2' if X.shape[1] > 1 else 'Feature 1')
        
        ax.set_title(f'{algo} Clustering Result', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_cluster_details(self, labels, n_clusters, noise):
        # Clear cluster info frame
        for widget in self.cluster_info_frame.winfo_children():
            widget.destroy()
        
        # Create text widget for details
        text_widget = tk.Text(self.cluster_info_frame, height=20, font=("Helvetica", 11))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        details = f"=== CLUSTERING DETAILS ===\n\n"
        details += f"Number of clusters found: {n_clusters}\n"
        details += f"Number of noise points: {noise}\n"
        details += f"Total points: {len(labels)}\n\n"
        
        # Count points per cluster
        unique, counts = np.unique(labels, return_counts=True)
        details += "Points per cluster:\n"
        
        for label, count in zip(unique, counts):
            if label == -1:
                details += f"  Noise: {count} points\n"
            else:
                percentage = (count / len(labels)) * 100
                details += f"  Cluster {label}: {count} points ({percentage:.1f}%)\n"
        
        text_widget.insert(tk.END, details)
        text_widget.config(state=tk.DISABLED)

    def export_labels(self):
        if not self.clustering_history:
            messagebox.showwarning("Warning", "No clustering results to export")
            return
        
        # Get the latest clustering result
        latest_result = self.clustering_history[-1]
        labels = latest_result['labels']
        
        # Create DataFrame with labels
        labels_df = pd.DataFrame({
            'cluster_label': labels
        })
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="cluster_labels.csv"
        )
        
        if file_path:
            try:
                labels_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Labels exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")




    # ------------------------
        #Classification Tab
    def build_classification_tab(self):
        main_frame = tk.Frame(self.classification_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.LabelFrame(
            main_frame,
            text="⚙️ Classification Settings",
            padding=10
        )
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        tk.Label(left_panel, text="Classification Algorithm:", 
                font=("Helvetica", 12, "bold"), bg="#ecf0f1").pack(pady=10, padx=10)

        self.class_algo_var = tk.StringVar(value="KNN")
        algo_list = ["KNN", "Decision Tree", "Naive Bayes", "SVM"]
        self.class_algo_combo = ttk.Combobox(left_panel, values=algo_list,
                                            state="readonly",
                                            textvariable=self.class_algo_var)
        self.class_algo_combo.pack(fill="x", padx=10)

        # Target column selection
        tk.Label(left_panel, text="Target column:",
                font=("Helvetica", 12, "bold"), bg="#ecf0f1").pack(pady=10, padx=10)

        self.target_var = tk.StringVar(value="")
        self.target_combo = ttk.Combobox(left_panel, state="readonly",
                                        textvariable=self.target_var)
        self.target_combo.pack(fill="x", padx=10)

        # Train button
        tk.Button(left_panel, text="🚀 Train Model", 
                bg=self.secondary_color, fg="white", font=("Helvetica", 12, "bold"),
                command=self.run_classification).pack(fill="x", padx=10, pady=20)

                # ===== Right panel (horizontal layout) =====
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ---------- LEFT: Results Table ----------
        table_frame = ttk.LabelFrame(right_panel, text="📋 Classification Results", padding=10)
        table_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.class_result_table = ttk.Treeview(
            table_frame,
            columns=("Algo", "Acc", "Target"),
            show="headings",
            height=12
        )

        self.class_result_table.heading("Algo", text="Algorithm")
        self.class_result_table.heading("Acc", text="Accuracy")
        self.class_result_table.heading("Target", text="Target Column")

        self.class_result_table.column("Algo", width=120, anchor="center")
        self.class_result_table.column("Acc", width=100, anchor="center")
        self.class_result_table.column("Target", width=150, anchor="center")

        self.class_result_table.pack(fill=tk.Y, expand=True)

        # ---------- RIGHT: Visualization ----------
        viz_frame = ttk.LabelFrame(right_panel, text="📊 Classification Visualization", padding=10)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.class_fig = plt.Figure(figsize=(6, 4))
        self.class_ax = self.class_fig.add_subplot(111)

        self.class_canvas = FigureCanvasTkAgg(self.class_fig, master=viz_frame)
        self.class_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
      
    # ------------------------
    # Fonction classification
    def run_classification(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No dataset loaded")
            return

        target = self.target_var.get()
        
        if target == "":
            messagebox.showwarning("Warning", "Select target column")
            return

        if target not in self.original_target:
            self.original_target[target] = self.data[target].copy()


        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC

        # Use preserved target (non-normalized)
        try: 
            y = self.original_target[target]
            validate_classification_target(y)
        except ValueError as e:
            messagebox.showerror("Classification Error", str(e))

        # Features only (numeric & normalized)
        X = self.data.drop(columns=[target])
        X = X.select_dtypes(include=[np.number])
        validate_no_categorical(X)
        validate_missing_values(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        algo = self.class_algo_var.get()

        if algo == "KNN":
            model = KNeighborsClassifier()
        elif algo == "Decision Tree":
            model = DecisionTreeClassifier()
        elif algo == "Naive Bayes":
            model = GaussianNB()
        elif algo == "SVM":
            model = SVC()
        else:
            messagebox.showwarning("Warning", "Unknown algorithm")
            return

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

       
        # Ajouter une ligne dans le tableau d'historique
        self.class_result_table.insert("", "end",values=(algo, f"{acc:.3f}", target))
        # ===== Confusion Matrix Visualization =====
        self.class_ax.clear()

        cm = confusion_matrix(y_test, preds)
        im = self.class_ax.imshow(cm)

        self.class_ax.set_title("Confusion Matrix")
        self.class_ax.set_xlabel("Predicted")
        self.class_ax.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.class_ax.text(j, i, cm[i, j],
                                ha="center", va="center")

        self.class_canvas.draw()

    def build_results_tab(self):
            main_frame = tk.Frame(self.results_tab)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Title
            title_label = tk.Label(main_frame, text="📈 Analysis & Results", 
                                font=("Helvetica", 20, "bold"), fg=self.primary_color)
            title_label.pack(pady=(0, 20))
            
            # Analysis buttons
            button_frame = tk.Frame(main_frame)
            button_frame.pack(fill='x', pady=10)
            
            tk.Button(button_frame, text="Show Dataset Summary", font=("Helvetica", 11),
                    bg=self.secondary_color, fg="white", 
                    command=self.show_summary_statistics).pack(side=tk.LEFT, padx=5)
            
            # Summary display area
            summary_frame = ttk.LabelFrame(main_frame, text="Dataset Summary", padding=10)
            summary_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            summary_scroll = tk.Scrollbar(summary_frame)
            summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.summary_text = tk.Text(summary_frame, height=20, font=("Courier", 10),
                                    yscrollcommand=summary_scroll.set, wrap=tk.WORD)
            self.summary_text.pack(fill=tk.BOTH, expand=True)
            summary_scroll.config(command=self.summary_text.yview)

        # ------------------------
        # Results Tab
        # ------------------------
    def build_results_tab(self):
            main_frame = tk.Frame(self.results_tab)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Title
            title_label = tk.Label(main_frame, text="📈 Analysis & Results", 
                                font=("Helvetica", 20, "bold"), fg=self.primary_color)
            title_label.pack(pady=(0, 20))
            
            # Analysis buttons
            button_frame = tk.Frame(main_frame)
            button_frame.pack(fill='x', pady=10)
            
            tk.Button(button_frame, text="Show Dataset Summary", font=("Helvetica", 11),
                    bg=self.secondary_color, fg="white", 
                    command=self.show_summary_statistics).pack(side=tk.LEFT, padx=5)
            
            # Summary display area
            summary_frame = ttk.LabelFrame(main_frame, text="Dataset Summary", padding=10)
            summary_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            summary_scroll = tk.Scrollbar(summary_frame)
            summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.summary_text = tk.Text(summary_frame, height=20, font=("Courier", 10),
                                    yscrollcommand=summary_scroll.set, wrap=tk.WORD)
            self.summary_text.pack(fill=tk.BOTH, expand=True)
            summary_scroll.config(command=self.summary_text.yview)

    def show_summary_statistics(self):
            if self.data is None:
                messagebox.showwarning("Warning", "No dataset loaded")
                return
            
            self.summary_text.delete(1.0, tk.END)
            
            summary = "=== DATASET SUMMARY ===\n\n"
            summary += f"Shape: {self.data.shape}\n"
            summary += f"Total entries: {len(self.data)}\n"
            summary += f"Total features: {len(self.data.columns)}\n\n"
            
            summary += "=== DATA TYPES ===\n"
            dtype_counts = self.data.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                summary += f"{dtype}: {count} columns\n"
            summary += "\n"
            
            summary += "=== MISSING VALUES ===\n"
            missing = self.data.isnull().sum()
            total_missing = missing.sum()
            summary += f"Total missing values: {total_missing}\n"
            if total_missing > 0:
                for col, count in missing[missing > 0].items():
                    percentage = (count / len(self.data)) * 100
                    summary += f"  {col}: {count} ({percentage:.1f}%)\n"
            summary += "\n"
            
            summary += "=== NUMERIC FEATURES SUMMARY ===\n"
            numeric_data = self.data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 0:
                summary += str(numeric_data.describe())
            else:
                summary += "No numeric features found\n"
            
            self.summary_text.insert(tk.END, summary)
    
    def show_preview(self):
            dataset_name = self.data_name.get()
            
            # Update all labels
            for label in [self.dataset_label_preview, self.dataset_label_preprocessing, 
                        self.dataset_label_clustering]:
                label.config(text=f"Dataset: {dataset_name}")
            
            if self.data is None:
                return
            
            # --- Head ---
            self.head_tree.delete(*self.head_tree.get_children())
            
            # Configure columns
            cols = list(self.data.columns)
            self.head_tree["columns"] = cols
            
            for col in cols:
                self.head_tree.heading(col, text=col[:20], anchor="center")
                self.head_tree.column(col, width=100, anchor="center")
            
            # Insert data (first 10 rows)
            for i, row in self.data.head(10).iterrows():
                formatted_row = []
                for val in row:
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        formatted_row.append(f"{val:.3f}")
                    else:
                        formatted_row.append(val)
                self.head_tree.insert("", "end", values=formatted_row)
            
            # --- Info ---
            self.info_tree.delete(*self.info_tree.get_children())
            for col in self.data.columns:
                dtype = str(self.data[col].dtype)
                missing = self.data[col].isnull().sum()
                self.info_tree.insert("", "end", values=(col, dtype, missing))
            
            # --- Stats ---
            self.stat_tree.delete(*self.stat_tree.get_children())
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                series = self.data[col].dropna()
                if len(series) > 0:
                    min_ = series.min()
                    q1 = series.quantile(0.25)
                    median = series.median()
                    q3 = series.quantile(0.75)
                    max_ = series.max()
                    mean = series.mean()
                    try:
                        mode = series.mode().iloc[0]
                    except:
                        mode = np.nan
                    
                    self.stat_tree.insert("", "end", 
                                        values=(col, f"{min_:.4f}", f"{q1:.4f}", 
                                                f"{median:.4f}", f"{q3:.4f}", 
                                                f"{max_:.4f}", f"{mean:.4f}", 
                                                f"{mode:.4f}" if not np.isnan(mode) else "N/A"))

            if hasattr(self, "target_combo"):
                self.target_combo["values"] = list(self.data.columns)
  
if __name__ == "__main__":
    app = DMIApp()
    app.mainloop()