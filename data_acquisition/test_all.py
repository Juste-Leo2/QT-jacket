import spidev
import time

# ==========================================
# --- CONFIGURATION SPI ---
# ==========================================
SPI_BUS = 0
SPI_DEVICE = 0
CANAUX = [0, 1, 2, 3, 4]   # les 5 canaux à lire

# Initialisation
spi = spidev.SpiDev()
spi.open(SPI_BUS, SPI_DEVICE)
spi.max_speed_hz = 1350000

# ==========================================
# --- FONCTION DE LECTURE (inchangée) ---
# ==========================================
def lire_canal_mcp3008(canal):
    """Lit un canal du MCP3008 et retourne la valeur brute (0-1023)."""
    if canal < 0 or canal > 7:
        return -1
    r = spi.xfer2([1, (8 + canal) << 4, 0])
    valeur = ((r[1] & 3) << 8) + r[2]
    return valeur

# ==========================================
# --- ACQUISITION CONTINUE SUR 5 CANAUX ---
# ==========================================
print("Acquisition continue sur les canaux 0 à 4 (Ctrl+C pour arrêter)")
print("CH0   CH1   CH2   CH3   CH4")

try:
    while True:
        # Lecture de chaque canal
        valeurs = [lire_canal_mcp3008(ch) for ch in CANAUX]
        
        # Formatage en une ligne avec valeurs brutes et tensions
        parties = []
        for ch, val in zip(CANAUX, valeurs):
            tension = (val * 3.3) / 1023.0
            parties.append(f"CH{ch}: {val:4d} (~{tension:.2f} V)")
        ligne = " | ".join(parties)
        
        print(ligne, end='\r')   # \r permet de rester sur la même ligne
        time.sleep(0.1)          # 10 lectures par seconde (ajustable)
        
except KeyboardInterrupt:
    print("\nArrêt demandé.")
finally:
    spi.close()
    print("SPI fermé.")
