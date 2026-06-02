import socket
import time
import collections
from collections import Counter
import numpy as np
import onnxruntime as ort
import sys
import os
import threading

# --- CONFIGURATION DU SERVEUR ---
HOST = '0.0.0.0'
PORT = 65432

# --- PARAMÈTRES IA / SIGNAL ---
ACQUISITION_FREQ = 1000       # 1kHz d'acquisition matérielle
DOWNSAMPLING = 10             # Moyenne tous les 10 points (-> 100Hz effectifs)
MODEL_PATH = "veste_model.onnx"

# Nos 6 classes telles que définies dans le README
CLASSES = [
    "Rien / Bruit", 
    "Tape_Attention", 
    "Caresse_Reconfortante", 
    "Chatouilles", 
    "Calin", 
    "Prise_Forte"
]

WINDOW_SIZE = 500             # 500 points à 100Hz = 5 secondes (identique à l'entraînement)
INFERENCE_STRIDE = 20          # 1 prédiction tous les 5 points (soit 20 prédictions par seconde)
VOTE_WINDOW_SIZE = 5          
SCORE_THRESHOLD = 0.80        
SEUIL_STABILITE = 0.5         # Écart-type minimum pour déclencher l'IA (à ajuster empiriquement)

# Paramètres de confirmation
CONFIRMATION_SIZE = 3         # 3 votes majoritaires consécutifs identiques pour valider un geste
COOLDOWN_DELAY = 5.0          # 5 secondes de pause après avoir détecté un geste

# --- CONFIGURATION MATÉRIELLE (SPI) ---
SPI_BUS = 0
SPI_DEVICE = 0
CANAUX = [4, 3, 0, 1, 2]      # Ordre des 5 capteurs (Bras G, Bras D, Torse G, Torse D, Dos)

SPIDEV_AVAILABLE = False
try:
    import spidev
    spi = spidev.SpiDev()
    spi.open(SPI_BUS, SPI_DEVICE)
    spi.max_speed_hz = 1350000
    SPIDEV_AVAILABLE = True
    print("Spidev initialisé avec succès pour l'inférence. Mode Réel.")
except Exception as e:
    print(f"ATTENTION: Module spidev non trouvé ({e}). Simulation activée.")

def read_adc(canal):
    """Lecture d'un canal MCP3008 via SPI"""
    if SPIDEV_AVAILABLE:
        r = spi.xfer2([1, (8 + canal) << 4, 0])
        val = ((r[1] & 3) << 8) + r[2]
        return val
    else:
        # Simulation d'un signal plat avec un peu de bruit
        return 512 + np.random.randint(-10, 10)


class GestureThread(threading.Thread):
    def __init__(self, client_socket=None):
        super().__init__()
        self.client_socket = client_socket
        self.running = True
        
        # --- CHARGEMENT DU MODÈLE ONNX ---
        print(f"[Thread] Chargement du modèle ONNX: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"ERREUR: {MODEL_PATH} introuvable. Avez-vous exécuté export_onnx.py ?")
            self.running = False
            return

        try:
            self.session = ort.InferenceSession(MODEL_PATH)
            self.input_name = self.session.get_inputs()[0].name
            print("[Thread] Modèle chargé avec succès.")
        except Exception as e:
            print(f"[Thread] Erreur ONNX: {e}")
            self.running = False

        # Buffers de données : chaque élément du raw_buffer contiendra 5 valeurs (les 5 capteurs)
        self.raw_buffer = collections.deque(maxlen=WINDOW_SIZE)
        self.vote_buffer = collections.deque(maxlen=VOTE_WINDOW_SIZE)
        self.confirmation_buffer = collections.deque(maxlen=CONFIRMATION_SIZE)
        
        # Accumulateur pour faire la moyenne de sous-échantillonnage
        self.accumulator = []

        # Initialisation "Intelligente" pour éviter de détecter un faux mouvement au démarrage
        self._reset_buffers(initial_fill=True)

    def _reset_buffers(self, initial_fill=False):
        """
        Vide tous les buffers.
        Si initial_fill=True, on lit les capteurs pour remplir le buffer avec l'état statique actuel,
        ce qui assure une dérivée nulle au démarrage.
        """
        self.vote_buffer.clear()
        for _ in range(VOTE_WINDOW_SIZE): 
            self.vote_buffer.append(0)
        
        self.confirmation_buffer.clear()
        self.accumulator = []
        
        if initial_fill:
            vals = []
            for _ in range(10):
                vals.append([read_adc(c) for c in CANAUX])
            # Moyenne des capteurs sur 10 lectures rapides
            avg_start = np.mean(vals, axis=0) 
            
            self.raw_buffer.clear()
            for _ in range(WINDOW_SIZE): 
                self.raw_buffer.append(avg_start)

    def run(self):
        print("[Thread] Démarrage de la boucle d'inférence temps réel...")
        points_processed_counter = 0 
        
        while self.running:
            try:
                loop_start = time.time()
                
                # 1. ACQUISITION (1kHz pour les 5 canaux)
                current_vals = [read_adc(c) for c in CANAUX]
                self.accumulator.append(current_vals)
                
                # 2. SOUS-ÉCHANTILLONNAGE (Vers 100Hz)
                if len(self.accumulator) >= DOWNSAMPLING:
                    # Moyenne sur 10 points le long des colonnes (axis=0)
                    avg_val = np.mean(self.accumulator, axis=0)
                    self.accumulator = []
                    
                    self.raw_buffer.append(avg_val)
                    points_processed_counter += 1
                    
                    # 3. INFÉRENCE (Une fois tous les INFERENCE_STRIDE points)
                    if points_processed_counter % INFERENCE_STRIDE == 0:
                        # Matrice de forme (500, 5)
                        raw_signal = np.array(self.raw_buffer, dtype=np.float32)
                        
                        ecart_type = np.std(raw_signal)
                        instant_pred = 0
                        
                        # Optimisation : On ne déclenche l'IA que si le signal n'est pas plat
                        if ecart_type >= SEUIL_STABILITE:
                            
                            # 3.1 Calcul de la dérivée (comme dans preprocess_data.py)
                            # Différence point par point sur l'axe du temps (axis=0)
                            derivative = np.diff(raw_signal, axis=0)
                            
                            # Ajout d'une ligne de zéros au début pour conserver la taille de 500 points
                            pad_zeros = np.zeros((1, 5), dtype=np.float32)
                            derivative = np.vstack((pad_zeros, derivative))
                            
                            # 3.2 Formatage pour le modèle PyTorch
                            # Actuellement: [500 points, 5 canaux]. Attendu: [1 batch, 5 canaux, 500 points]
                            input_data = derivative.transpose(1, 0)
                            input_data = input_data.reshape(1, 5, WINDOW_SIZE)
                            
                            # 3.3 Lancement ONNX
                            outputs = self.session.run(None, {self.input_name: input_data})
                            
                            # 3.4 Softmax pour obtenir les probabilités (0 à 1)
                            exp_preds = np.exp(outputs[0][0])
                            probs = exp_preds / np.sum(exp_preds)
                            
                            if np.max(probs) > SCORE_THRESHOLD:
                                instant_pred = np.argmax(probs)
                        
                        # --- LOGIQUE DE VOTE ---
                        # 1. Vote court terme (Lissage des prédictions instantanées)
                        self.vote_buffer.append(instant_pred)
                        winner_class, _ = Counter(self.vote_buffer).most_common(1)[0]
                        
                        # 2. Vote de confirmation (Empêche les faux positifs sporadiques)
                        self.confirmation_buffer.append(winner_class)

                        if len(self.confirmation_buffer) == CONFIRMATION_SIZE:
                            unique_votes = set(self.confirmation_buffer)
                            
                            # Si les N derniers votes lissés sont IDENTIQUES et NON NULS
                            if len(unique_votes) == 1:
                                confirmed_gesture = list(unique_votes)[0]
                                
                                if confirmed_gesture != 0:
                                    nom = CLASSES[confirmed_gesture]
                                    print(f">>> GESTE DÉTECTÉ ET VALIDÉ : {nom} <<<")
                                    
                                    # Envoi au PC via Socket s'il est connecté
                                    if self.client_socket is not None:
                                        try:
                                            self.client_socket.sendall((nom + '\n').encode('utf-8'))
                                        except (BrokenPipeError, ConnectionResetError):
                                            print("[Thread] Connexion PC perdue.")
                                            self.client_socket = None
                                    
                                    # --- PAUSE RÉFRACTAIRE ---
                                    # On bloque l'analyse pour ne pas redétecter le même geste 10 fois
                                    print(f"[Thread] Pause de {COOLDOWN_DELAY}s (Stabilisation)...")
                                    time.sleep(COOLDOWN_DELAY)
                                    
                                    # --- RESET COMPLET ---
                                    print("[Thread] Reprise de l'écoute.")
                                    self._reset_buffers(initial_fill=True)
                                    continue 

                # 4. CADENCEMENT STRICT (1kHz)
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / ACQUISITION_FREQ) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            except Exception as e:
                print(f"[Thread] Erreur fatale boucle: {e}")
                self.running = False

    def stop(self):
        self.running = False


# --- POINT D'ENTRÉE DU SCRIPT ---
if __name__ == "__main__":
    print(f"--- Serveur d'inférence Veste (PID: {os.getpid()}) ---")
    sys.stdout.flush()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    client = None
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        
        # TIMEOUT TRÈS IMPORTANT : Le serveur n'attendra qu'une seconde.
        # S'il n'y a pas de PC qui s'y connecte, il passera quand même à la suite
        # et fonctionnera en mode Autonome (parfait pour le Raspberry Pi seul).
        server_socket.settimeout(1.0) 
        
        print(f"En attente de connexion du PC sur {HOST}:{PORT} (Facultatif)...")
        sys.stdout.flush()
        
        try:
            client, addr = server_socket.accept()
            print(f"--- PC Hôte Connecté : {addr} ---")
            client.settimeout(None) # Remettre le client en mode bloquant standard
        except socket.timeout:
            print("--- Aucun PC connecté. Lancement en mode AUTONOME. ---")
        
        # On lance le thread d'inférence (en lui passant le client si on en a un, ou None)
        gesture_thread = GestureThread(client)
        gesture_thread.start()
        
        # Le thread principal (main) attend juste que le sous-thread se termine
        while gesture_thread.is_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nArrêt manuel demandé par l'utilisateur.")
    except Exception as e:
        print(f"Erreur globale dans le Main: {e}")
    finally:
        print("Nettoyage et fermeture des connexions...")
        if 'gesture_thread' in locals():
            gesture_thread.stop()
            gesture_thread.join()
        if client is not None:
            client.close()
        server_socket.close()
