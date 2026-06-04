import os
import glob
import time
import threading
import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# --- HARDWARE CONFIGURATION ---
# ==========================================
SPI_BUS = 0
SPI_DEVICE = 0
CANAUX = [4, 3, 0, 1, 2]  # 5 canaux (Bras G, Bras D, Torse G, Torse D, Dos)

SPIDEV_AVAILABLE = False
try:
    import spidev
    spi = spidev.SpiDev()
    spi.open(SPI_BUS, SPI_DEVICE)
    spi.max_speed_hz = 1350000
    SPIDEV_AVAILABLE = True
    print("Spidev initialisé avec succès. Mode réel.")
except ImportError:
    print("Module spidev non trouvé. Lancement en mode SIMULATION.")
except FileNotFoundError:
    print("Périphérique SPI introuvable. Lancement en mode SIMULATION.")
except Exception as e:
    print(f"Erreur SPI ({e}). Lancement en mode SIMULATION.")

# ==========================================
# --- CLASSES & CONSTANTS ---
# ==========================================
CLASSES = {
    0: "Rien / Bruit de fond",
    1: "Tapotement Attention",
    2: "Caresse Réconfortante",
    3: "Chatouilles",
    4: "Étreinte / Câlin",
    5: "Agrippement Fort"
}

SAMPLING_RATE = 1000  # Hz
DURATION = 5  # seconds
TOTAL_POINTS = SAMPLING_RATE * DURATION
DATA_DIR = "data_collected"

# ==========================================
# --- UTILS ---
# ==========================================
def value_to_color(val):
    """Convert a 0-1023 value to a color (green to red)."""
    # Clamp value
    val = max(0, min(1023, val))
    # Normalize
    ratio = val / 1023.0
    r = int(255 * ratio)
    g = int(255 * (1 - ratio))
    b = 0
    return f'#{r:02x}{g:02x}{b:02x}'

# ==========================================
# --- MAIN APPLICATION ---
# ==========================================
class AcquisitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Data Acquisition - Smart Jacket")
        self.geometry("1100x700")
        
        # State
        self.current_class = None
        self.is_recording = False
        self.is_waiting = False
        self.temp_data = None
        self.running = True
        self.recording_buffer = np.zeros((TOTAL_POINTS, 5))
        self.recording_index = 0
        self.read_counter = 0
        self.class_counts = {k: 0 for k in CLASSES.keys()}
        
        self.setup_data_folders()
        self.count_existing_data()

        self.build_ui()
        threading.Thread(target=self.continuous_task, daemon=True).start()
        
    def setup_data_folders(self):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

    def count_existing_data(self):
        for class_id in CLASSES.keys():
            path = os.path.join(DATA_DIR, f"class_{class_id}.npz")
            if os.path.exists(path):
                with np.load(path) as data:
                    self.class_counts[class_id] = len(data.files)
            else:
                self.class_counts[class_id] = 0

    def build_ui(self):
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)

        # --- LEFT PANEL (VISUAL & GRAPH) ---
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.left_frame.grid_rowconfigure(0, weight=1) # Visual
        self.left_frame.grid_rowconfigure(1, weight=1) # Graph
        self.left_frame.grid_columnconfigure(0, weight=1)

        # VISUAL (Jacket)
        self.visual_frame = ctk.CTkFrame(self.left_frame)
        self.visual_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.visual_label = ctk.CTkLabel(self.visual_frame, text="Visuel de la Veste (Capteurs)", font=("Arial", 16, "bold"))
        self.visual_label.pack(pady=5)
        
        self.canvas = ctk.CTkCanvas(self.visual_frame, width=300, height=250, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        # Draw a simple jacket (rectangle for body, polygons for arms)
        self.canvas.create_rectangle(100, 50, 200, 200, fill="#3a3a3a", outline="white", width=2) # Torse
        self.canvas.create_polygon(100, 50, 50, 120, 80, 140, 100, 90, fill="#3a3a3a", outline="white", width=2) # Bras Gauche
        self.canvas.create_polygon(200, 50, 250, 120, 220, 140, 200, 90, fill="#3a3a3a", outline="white", width=2) # Bras Droit
        
        # Sensors (Circles)
        r = 12
        # C0: Bras G
        self.s0 = self.canvas.create_oval(60-r, 120-r, 60+r, 120+r, fill="green", outline="white")
        # C1: Bras D
        self.s1 = self.canvas.create_oval(240-r, 120-r, 240+r, 120+r, fill="green", outline="white")
        # C2: Torse Gauche
        self.s2 = self.canvas.create_oval(130-r, 100-r, 130+r, 100+r, fill="green", outline="white")
        # C3: Torse Droit
        self.s3 = self.canvas.create_oval(170-r, 100-r, 170+r, 100+r, fill="green", outline="white")
        # C4: Dos (affiché au centre bas)
        self.s4 = self.canvas.create_oval(150-r, 160-r, 150+r, 160+r, fill="green", outline="white")
        
        self.sensor_shapes = [self.s0, self.s1, self.s2, self.s3, self.s4]
        self.canvas.create_text(60, 140, text="Bras G", fill="white", font=("Arial", 9))
        self.canvas.create_text(240, 140, text="Bras D", fill="white", font=("Arial", 9))
        self.canvas.create_text(130, 80, text="Torse G", fill="white", font=("Arial", 9))
        self.canvas.create_text(170, 80, text="Torse D", fill="white", font=("Arial", 9))
        self.canvas.create_text(150, 180, text="Dos", fill="white", font=("Arial", 9))

        # GRAPH (Matplotlib)
        self.graph_frame = ctk.CTkFrame(self.left_frame)
        self.graph_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.fig, self.ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')
        self.ax.set_title("Aperçu des signaux bruts", color="white")
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        # --- RIGHT PANEL (CONTROLS & STATS) ---
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.status_label = ctk.CTkLabel(self.right_frame, text="Prêt", font=("Arial", 20, "bold"), text_color="green")
        self.status_label.pack(pady=20)

        # Class Buttons
        ctk.CTkLabel(self.right_frame, text="Lancer l'acquisition (5s)", font=("Arial", 14, "bold")).pack(pady=5)
        self.buttons_frame = ctk.CTkScrollableFrame(self.right_frame, height=150)
        self.buttons_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.class_buttons = []
        for class_id, class_name in CLASSES.items():
            btn = ctk.CTkButton(self.buttons_frame, text=f"Classe {class_id} : {class_name}", 
                                command=lambda c=class_id: self.start_acquisition(c))
            btn.pack(pady=5, fill="x", padx=20)
            self.class_buttons.append(btn)

        # Validation Buttons (Hidden initially)
        self.validation_frame = ctk.CTkFrame(self.right_frame)
        self.validation_frame.pack(fill="x", padx=10, pady=10)
        
        self.btn_valider = ctk.CTkButton(self.validation_frame, text="Valider & Sauvegarder", fg_color="green", hover_color="darkgreen", state="disabled", command=self.save_data)
        self.btn_valider.pack(side="left", padx=10, pady=10, expand=True)
        
        self.btn_rejeter = ctk.CTkButton(self.validation_frame, text="Rejeter", fg_color="red", hover_color="darkred", state="disabled", command=self.reject_data)
        self.btn_rejeter.pack(side="right", padx=10, pady=10, expand=True)

        # Stats
        self.stats_frame = ctk.CTkScrollableFrame(self.right_frame, height=150)
        self.stats_frame.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(self.stats_frame, text="Progression", font=("Arial", 14, "bold")).pack(pady=5)
        
        self.counter_labels = {}
        for class_id, class_name in CLASSES.items():
            lbl = ctk.CTkLabel(self.stats_frame, text=f"Classe {class_id}: {self.class_counts[class_id]} / 120")
            lbl.pack(pady=2)
            self.counter_labels[class_id] = lbl

    def update_stats_ui(self):
        for class_id in CLASSES.keys():
            self.counter_labels[class_id].configure(text=f"Classe {class_id}: {self.class_counts[class_id]} / 120")

    def toggle_buttons(self, state):
        mode = "normal" if state else "disabled"
        for btn in self.class_buttons:
            btn.configure(state=mode)

    def update_sensors_visual(self, values):
        """Update colors on the canvas based on sensor values (0-1023)"""
        for i, val in enumerate(values):
            color = value_to_color(val)
            self.canvas.itemconfig(self.sensor_shapes[i], fill=color)

    def start_acquisition(self, class_id):
        if self.is_recording or self.is_waiting:
            return
            
        self.current_class = class_id
        self.is_waiting = True
        self.toggle_buttons(False)
        self.btn_valider.configure(state="disabled")
        self.btn_rejeter.configure(state="disabled")
        
        self.status_label.configure(text=f"Préparation (1s)...", text_color="yellow")
        self.ax.clear()
        self.ax.set_title("Aperçu des signaux bruts", color="white")
        self.canvas_plot.draw()

        self.after(1000, self.begin_recording)

    def begin_recording(self):
        self.is_waiting = False
        self.is_recording = True
        self.recording_index = 0
        self.status_label.configure(text=f"Acquisition Classe {self.current_class}...", text_color="orange")

    def lire_mcp3008(self, canal):
        if not SPIDEV_AVAILABLE:
            return 0
        r = spi.xfer2([1, (8 + canal) << 4, 0])
        valeur = ((r[1] & 3) << 8) + r[2]
        return valeur

    def continuous_task(self):
        sim_phase = 0.0
        target_time = time.time()
        
        while self.running:
            row = []
            if SPIDEV_AVAILABLE:
                for ch in CANAUX:
                    row.append(self.lire_mcp3008(ch))
            else:
                # SIMULATION
                noise = np.random.randint(0, 50, 5)
                pattern = np.zeros(5)
                if self.is_recording:
                    if self.current_class == 1:
                        if 1000 < self.recording_index < 1200: pattern[0] = 600
                    elif self.current_class == 2:
                        if 500 < self.recording_index < 4500: pattern[2] = 300 + 100*np.sin(sim_phase)
                    elif self.current_class == 5:
                        if 1000 < self.recording_index < 4000: pattern = np.array([800, 800, 800, 800, 800])
                
                sim_phase += 0.01
                row = np.clip(noise + pattern, 0, 1023)
                
            self.read_counter += 1
            if self.read_counter % 50 == 0:
                self.after(0, self.update_sensors_visual, row)
                
            if self.is_recording:
                self.recording_buffer[self.recording_index] = row
                self.recording_index += 1
                if self.recording_index >= TOTAL_POINTS:
                    self.is_recording = False
                    self.temp_data = np.copy(self.recording_buffer)
                    self.after(0, self.post_acquisition)
            
            target_time += (1.0 / SAMPLING_RATE)
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def post_acquisition(self):
        self.is_recording = False
        self.status_label.configure(text="Terminé. Valider ou Rejeter ?", text_color="cyan")
        
        # Reset sensors visually
        self.update_sensors_visual([0, 0, 0, 0, 0])
        
        # Plot data
        self.ax.clear()
        time_axis = np.linspace(0, DURATION, TOTAL_POINTS)
        labels = ["Bras G", "Bras D", "Torse G", "Torse D", "Dos"]
        
        for i in range(5):
            self.ax.plot(time_axis, self.temp_data[:, i], label=labels[i])
            
        self.ax.legend(loc="upper right", fontsize='small')
        self.ax.set_title(f"Acquisition Classe {self.current_class}", color="white")
        self.ax.set_xlabel("Temps (s)", color="white")
        self.ax.set_ylabel("Valeur brute", color="white")
        self.canvas_plot.draw()

        # Enable validation buttons
        self.btn_valider.configure(state="normal")
        self.btn_rejeter.configure(state="normal")

    def save_data(self):
        if self.temp_data is None:
            return
            
        class_id = self.current_class
        path = os.path.join(DATA_DIR, f"class_{class_id}.npz")
        
        # Load existing data if any
        data_dict = {}
        if os.path.exists(path):
            with np.load(path) as existing:
                for k in existing.files:
                    data_dict[k] = existing[k]
                    
        # Append new sample
        next_idx = len(data_dict) + 1
        data_dict[f"sample_{next_idx:03d}"] = self.temp_data
        
        # Save back
        np.savez(path, **data_dict)
        
        # Update UI
        self.class_counts[class_id] += 1
        self.update_stats_ui()
        
        self.reset_state("Donnée sauvegardée ! Prêt.")

    def reject_data(self):
        self.reset_state("Donnée rejetée. Prêt.")

    def reset_state(self, message):
        self.temp_data = None
        self.current_class = None
        self.is_waiting = False
        self.btn_valider.configure(state="disabled")
        self.btn_rejeter.configure(state="disabled")
        self.toggle_buttons(True)
        self.status_label.configure(text=message, text_color="green")
        self.ax.clear()
        self.ax.set_title("Aperçu des signaux bruts", color="white")
        self.canvas_plot.draw()

if __name__ == "__main__":
    # CustomTkinter Appearance
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    
    app = AcquisitionApp()
    app.mainloop()
