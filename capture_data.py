import spidev
import time
import os
import numpy as np

# ==========================================
# --- CONFIGURATION ---
# ==========================================
# IMPORTANT : Pour utiliser spidev, vous devez connecter le MCP3008 aux pins SPI matériels du Raspberry Pi :
# VDD  -> 3.3V
# VREF -> 3.3V
# AGND -> GND
# DGND -> GND
# CLK  -> SCLK (Pin 23 / GPIO 11)
# DOUT -> MISO (Pin 21 / GPIO 9)
# DIN  -> MOSI (Pin 19 / GPIO 10)
# CS   -> CE0  (Pin 24 / GPIO 8)

SPI_BUS = 0
SPI_DEVICE = 0
CANAUX_ACTIFS = [0, 1, 2, 3] # Liste des canaux à lire (ex: 4 capteurs sur CH0 à CH3)

FILENAME = "dataset.npz"
SAMPLES_PER_WINDOW = 100  # Nombre de points par enregistrement (ex: 100 points)
TARGET_FREQ = 100         # Fréquence de capture (100 Hz = 100 points par seconde)

CLASSES = {0: "RIEN", 1: "TAPE", 2: "PINCEMENT", 3: "FROTTEMENT"}

# ==========================================
# --- INITIALISATION SPI ---
# ==========================================
spi = spidev.SpiDev()
spi.open(SPI_BUS, SPI_DEVICE)
spi.max_speed_hz = 1350000 # Fréquence de communication SPI (1.35 MHz)

# ==========================================
# --- FONCTIONS UTILITAIRES ---
# ==========================================

def lire_canal_mcp3008(canal):
    """
    Lit un canal du MCP3008 via le bus SPI.
    Retourne une valeur numérique brute entre 0 et 1023.
    """
    if canal < 0 or canal > 7:
        return -1
    
    # On envoie la commande de lecture pour le canal choisi
    # Le format de la trame SPI pour le MCP3008 est spécifique (voir datasheet)
    r = spi.xfer2([1, (8 + canal) << 4, 0])
    
    # On reconstruit le résultat sur 10 bits à partir des octets reçus
    valeur = ((r[1] & 3) << 8) + r[2]
    return valeur

def charger_dataset():
    """Charge les données existantes du fichier .npz s'il existe."""
    if os.path.exists(FILENAME):
        try:
            data = np.load(FILENAME)
            # On retourne sous forme de listes pour pouvoir ajouter des éléments facilement
            return list(data['X']), list(data['y'])
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
            return [], []
    return [], []

def sauvegarder_dataset(X, y):
    """Sauvegarde les données dans un fichier .npz compréssé, optimisé pour le Machine Learning."""
    # X devient un tableau 3D de forme : (Nombre_echantillons, Temps, Canaux)
    # y devient un tableau 1D de forme : (Nombre_echantillons,)
    np.savez_compressed(FILENAME, X=np.array(X), y=np.array(y))
    print(f"💾 Sauvegarde réussie dans {FILENAME}")

def ascii_plot(echantillon):
    """
    Affiche une petite visualisation dans le terminal.
    Pour ne pas surcharger l'écran, on n'affiche que le premier capteur.
    """
    # echantillon contient toutes les mesures (Temps, Canaux)
    # On extrait juste la liste des valeurs du premier canal
    canal_0 = [point_temporel[0] for point_temporel in echantillon]
    
    mini = min(canal_0)
    maxi = max(canal_0)
    
    print(f"Capteur 1 (CH0) -> Min: {mini} | Max: {maxi}")
    
    # Si la variation est trop faible, c'est que le signal est plat
    if maxi - mini < 5: 
        print("[----- Signal Plat (Stable) -----]")
        return
        
    # On affiche 1 point sur 5 pour l'animation (20 lignes pour 100 points)
    for val in canal_0[::5]: 
        # Produit en croix pour dessiner maximum 50 étoiles
        stars = int(((val - mini) / (maxi - mini + 1)) * 50)
        print("|" + "*" * stars)

def enregistrer_batch(label_id, nombre, X_global, y_global):
    """Effectue une série d'enregistrements et les stocke en mémoire."""
    print(f"\n>>> ENREGISTREMENT : {CLASSES[label_id]} <<<")
    
    for i in range(nombre):
        print(f"\n--- Prise {i+1}/{nombre} ---")
        
        # Petit compte à rebours pour se préparer, sauf si c'est du bruit de fond
        if label_id != 0:
            for x in range(3, 0, -1):
                print(f"{x}...", end=' ', flush=True)
                time.sleep(0.5)
            print("GO !")
        else:
            print("Enregistrement bruit/repos...")
        
        echantillon = [] # Contiendra les 100 pas de temps
        
        # --- BOUCLE PRINCIPALE D'ACQUISITION ---
        for _ in range(SAMPLES_PER_WINDOW):
            point_start = time.time()
            
            valeurs_temporelles = []
            
            # On lit tous nos capteurs actifs, un par un
            for canal in CANAUX_ACTIFS:
                val = lire_canal_mcp3008(canal)
                valeurs_temporelles.append(val)
                
            # On ajoute ces mesures au temps T à l'échantillon
            echantillon.append(valeurs_temporelles)
            
            # --- GESTION DU TEMPS (CADENCEMENT) ---
            # On calcule combien de temps a pris la lecture
            elapsed = time.time() - point_start
            sleep_time = (1.0 / TARGET_FREQ) - elapsed
            
            # On patiente le temps restant pour atteindre la fréquence désirée
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Une fois l'échantillon terminé, on l'ajoute au grand jeu de données
        X_global.append(echantillon)
        y_global.append(label_id)
        
        # Visualisation
        ascii_plot(echantillon)
        time.sleep(0.5)

    print(f"\n✅ {nombre} échantillons ajoutés en mémoire.")
    # On sauvegarde sur le disque après chaque série pour sécuriser les données
    sauvegarder_dataset(X_global, y_global)

# ==========================================
# --- MENU PRINCIPAL ---
# ==========================================

# On charge les anciennes données si elles existent
X_data, y_data = charger_dataset()
if len(X_data) > 0:
    print(f"📁 Dataset existant chargé : {len(X_data)} échantillons trouvés.")

try:
    while True:
        print("\n" + "="*40)
        print(f"ACQUISITION DE DONNÉES (Fichier: {FILENAME})")
        print(f"Échantillons en mémoire : {len(X_data)}")
        print("0: RIEN")
        print("1: TAPE")
        print("2: PINCEMENT")
        print("3: FROTTEMENT")
        print("Q: Quitter")
        
        choice = input("Choix : ").upper()
        
        if choice == 'Q':
            print("Sauvegarde finale avant de quitter...")
            sauvegarder_dataset(X_data, y_data)
            break
        
        if choice in ['0', '1', '2', '3']:
            try:
                label = int(choice)
                nb = int(input(f"Nombre d'échantillons pour {CLASSES[label]} : "))
                enregistrer_batch(label, nb, X_data, y_data)
            except ValueError:
                print("⚠️ Veuillez entrer un nombre valide.")
        else:
            print("⚠️ Choix inconnu.")

except KeyboardInterrupt:
    # Si l'utilisateur fait Ctrl+C, on sauvegarde quand même !
    print("\nInterruption détectée. Sauvegarde d'urgence...")
    sauvegarder_dataset(X_data, y_data)
    print("Bye !")
finally:
    spi.close()
