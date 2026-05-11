import spidev
import time

# ==========================================
# --- CONFIGURATION SPI (identique au script original) ---
# ==========================================
SPI_BUS = 0
SPI_DEVICE = 0
CANAL_TEST = 0          # On ne lit que le canal 0

# Initialisation
spi = spidev.SpiDev()
spi.open(SPI_BUS, SPI_DEVICE)
spi.max_speed_hz = 1350000

# ==========================================
# --- FONCTION DE LECTURE ---
# ==========================================
def lire_canal_mcp3008(canal):
    """Lit un canal du MCP3008 et retourne la valeur brute (0-1023)."""
    if canal < 0 or canal > 7:
        return -1
    r = spi.xfer2([1, (8 + canal) << 4, 0])
    valeur = ((r[1] & 3) << 8) + r[2]
    return valeur

# ==========================================
# --- ACQUISITION CONTINUE SIMPLE ---
# ==========================================
print("Acquisition continue sur le canal 0 (Ctrl+C pour arrêter)")
print("Valeur brute (0-1023) :")

try:
    while True:
        valeur = lire_canal_mcp3008(CANAL_TEST)
        # Option : afficher aussi la tension approximative
        tension = (valeur * 3.3) / 1023.0
        print(f"CH0: {valeur:4d}  |  ~{tension:.2f} V", end='\r')  # \r pour rester sur la même ligne
        time.sleep(0.1)   # 10 lectures par seconde (ajustez selon besoin)
except KeyboardInterrupt:
    print("\nArrêt demandé.")
finally:
    spi.close()
    print("SPI fermé.")
