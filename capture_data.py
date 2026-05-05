from gpiozero import MCP3008
import time
import csv
import os
import sys
import numpy as np # On utilise numpy pour la moyenne, c'est plus propre

# --- CONFIGURATION ---
adc = MCP3008(channel=0, 
              clock_pin=12, 
              mosi_pin=20, 
              miso_pin=16, 
              select_pin=21)

FILENAME = "real_dataset.csv"
SAMPLES_PER_WINDOW = 100  # 1 seconde de données utiles
TARGET_FREQ = 100         # Fréquence de sortie (100 Hz)
OVERSAMPLING = 10         # Nombre de lectures pour faire 1 point (donc lecture brute à 1 kHz)

CLASSES = {0: "RIEN", 1: "TAPE", 2: "PINCEMENT", 3: "FROTTEMENT"}

def init_csv():
    if not os.path.exists(FILENAME):
        with open(FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [str(i) for i in range(SAMPLES_PER_WINDOW)] + ['label']
            writer.writerow(header)

def ascii_plot(data):
    mini = min(data)
    maxi = max(data)
    print(f"Min: {mini:.3f} | Max: {maxi:.3f}")
    if maxi - mini < 0.03:
        print("[----- Signal Plat (Stable) -----]")
        return
    for val in data[::5]: 
        stars = int(val * 50)
        print("|" + "*" * stars)

def record_batch(label_id, count):
    print(f"\n>>> ENREGISTREMENT : {CLASSES[label_id]} (Mode Moyenneur 1kHz -> 100Hz) <<<")
    
    with open(FILENAME, 'a', newline='') as f:
        writer = csv.writer(f)
        
        for i in range(count):
            print(f"\n--- Prise {i+1}/{count} ---")
            
            if label_id != 0:
                for x in range(3, 0, -1):
                    print(f"{x}...", end=' ', flush=True)
                    time.sleep(0.5)
                print("GO !")
            else:
                print("Enregistrement bruit/repos...")
            
            buffer = []
            
            # Boucle principale (100 points à générer)
            for _ in range(SAMPLES_PER_WINDOW):
                point_start = time.time()
                
                # --- SURÉCHANTILLONNAGE (Oversampling) ---
                # On lit 10 valeurs très vite
                raw_reads = []
                for _ in range(OVERSAMPLING):
                    raw_reads.append(adc.value)
                    # Pas de sleep ici, on lit aussi vite que le Pi peut (ce sera ~1-5 kHz)
                
                # On fait la moyenne
                avg_val = sum(raw_reads) / len(raw_reads)
                buffer.append(f"{avg_val:.5f}") # 5 décimales pour la précision
                
                # --- CADENCEMENT ---
                # On s'assure que ce bloc "Lecture + Moyenne" a pris 10ms
                elapsed = time.time() - point_start
                sleep_time = (1.0 / TARGET_FREQ) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Sauvegarde
            row = buffer + [label_id]
            writer.writerow(row)
            
            # Visu
            float_data = [float(x) for x in buffer]
            ascii_plot(float_data)
            
            time.sleep(0.5)

    print(f"\n✅ {count} échantillons ajoutés.")

# --- MENU ---
init_csv()

try:
    while True:
        print("\n" + "="*40)
        print(f"ACQUISITION DE DONNÉES (Fichier: {FILENAME})")
        print("0: RIEN")
        print("1: TAPE")
        print("2: PINCEMENT")
        print("3: FROTTEMENT")
        print("Q: Quitter")
        
        choice = input("Choix : ").upper()
        
        if choice == 'Q':
            break
        
        if choice in ['0', '1', '2', '3']:
            try:
                nb = int(input(f"Nombre d'échantillons pour {CLASSES[int(choice)]} : "))
                record_batch(int(choice), nb)
            except ValueError:
                pass
        else:
            print("?")

except KeyboardInterrupt:
    print("\nBye !")
