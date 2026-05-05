# Tutoriel : Acquisition de données multi-capteurs (QT-Jacket)

Ce tutoriel explique comment utiliser le script `capture_data.py` pour enregistrer les données de plusieurs capteurs de votre veste vers un format adapté à l'Intelligence Artificielle (Machine Learning).

## 1. Prérequis

Pour que le script fonctionne de manière fluide et rapide, nous avons abandonné `gpiozero` au profit de `spidev`, qui communique directement avec le matériel du Raspberry Pi.

### Activer le port SPI
Si ce n'est pas déjà fait, vous devez activer l'interface SPI de votre Raspberry Pi :
1. Dans le terminal, tapez : `sudo raspi-config`
2. Allez dans `Interfacing Options` -> `SPI` -> Sélectionnez `Yes`.
3. Redémarrez le Raspberry Pi (`sudo reboot`).

### Installer les bibliothèques
Assurez-vous d'avoir les bibliothèques nécessaires. Sur les Raspberry Pi récents, il est recommandé d'utiliser `apt` :
```bash
sudo apt-get update
sudo apt-get install python3-numpy python3-spidev
```

## 2. Câblage du MCP3008 (Très Important !)

Contrairement à l'ancienne méthode, `spidev` **exige** que le MCP3008 soit branché sur les broches SPI officielles du Raspberry Pi. 
Voici le câblage à respecter :

* **VDD** et **VREF** se branchent sur **3.3V**
* **AGND** et **DGND** se branchent sur **GND** (Terre)
* **CLK** se branche sur la broche **SCLK** (Pin 23 / GPIO 11)
* **DOUT** se branche sur la broche **MISO** (Pin 21 / GPIO 9)
* **DIN** se branche sur la broche **MOSI** (Pin 19 / GPIO 10)
* **CS / SHDN** se branche sur la broche **CE0** (Pin 24 / GPIO 8)

Vos capteurs (piézoélectriques ou piézorésistifs) se branchent ensuite sur les canaux `CH0`, `CH1`, `CH2`, etc., du MCP3008.

## 3. Utilisation du script

Lancez le script depuis le terminal :
```bash
python capture_data.py
```

Un menu s'affiche. Le fonctionnement est le suivant :
1. Choisissez une action (ex: `1` pour TAPE).
2. Entrez le nombre d'échantillons que vous voulez faire (ex: `10`).
3. Le script compte "3... 2... 1... GO !". Faites le geste correspondant sur la veste.
4. L'enregistrement dure **1 seconde** (c'est la constante `SAMPLES_PER_WINDOW`).
5. Les données sont automatiquement sauvegardées sur la carte SD après chaque lot.
6. Appuyez sur `Q` pour quitter proprement.

## 4. Comprendre les données sauvegardées (.npz)

Au lieu d'un fichier CSV classique (qui deviendrait illisible avec plusieurs capteurs), le script crée un fichier `dataset.npz`. C'est un format de la bibliothèque `numpy`, très utilisé en Intelligence Artificielle.

**Comment relire ces données dans un autre script Python ?**

Voici un petit exemple de code pour charger vos données plus tard :

```python
import numpy as np

# Charger le fichier
donnees = np.load("dataset.npz")

# Récupérer les matrices
X = donnees['X'] # Les mesures des capteurs
y = donnees['y'] # Les labels (0=Rien, 1=Tape, etc.)

print(f"J'ai chargé {len(X)} enregistrements !")

# X est un tableau en 3 dimensions : (Enregistrement, Temps, Capteur)
# Par exemple, pour voir la mesure du Capteur 2, au pas de temps 50,
# du tout premier enregistrement (index 0) :
mesure = X[0, 50, 2]
print(f"Valeur brute mesurée : {mesure}")
```

Grâce à cette structure, vos données sont prêtes à être injectées dans un réseau de neurones !
