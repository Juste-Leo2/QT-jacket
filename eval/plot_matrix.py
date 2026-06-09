import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Labels tirés du README
CLASS_LABELS = ['Nothing / Noise', 'Tap Attention', 'Comforting Caress', 'Tickles', 'Hug / Cuddle']

def plot_confusion_matrices():
    # Définition des fichiers attendus (Numpy arrays sauvegardés par train.py)
    configs = {
        "Classique - Sans Dropout": "cm_tactile_raw.npy",
        "Dérivé - Sans Dropout": "cm_tactile_deriv.npy",
        "10 Canaux - Sans Dropout": "cm_tactile_10ch.npy",
        "Classique - Avec Dropout": "cm_tactile_raw_drop.npy",
        "Dérivé - Avec Dropout": "cm_tactile_deriv_drop.npy",
        "10 Canaux - Avec Dropout": "cm_tactile_10ch_drop.npy"
    }

    # Création d'une figure avec 2 lignes et 3 colonnes
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (title, filename) in enumerate(configs.items()):
        ax = axes[idx]
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        if os.path.exists(filepath):
            # Charger la matrice de confusion
            cm = np.load(filepath)
            
            # Afficher la matrice
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
            # Utilisation de différentes couleurs pour distinguer avec/sans dropout (optionnel)
            cmap = plt.cm.Blues if "Sans Dropout" in title else plt.cm.Oranges
            disp.plot(ax=ax, cmap=cmap, colorbar=False)
            
            # Tourner légèrement les labels sur l'axe X pour que le texte soit lisible
            ax.set_xticklabels(CLASS_LABELS, rotation=45, ha='right')
            ax.set_title(title, fontsize=12)
        else:
            # Si le fichier n'existe pas, masquer l'axe ou afficher un message
            ax.text(0.5, 0.5, f'Fichier introuvable:\n{filename}', 
                    ha='center', va='center', fontsize=10, color='gray')
            ax.axis('off')

    plt.suptitle("Comparaison des Matrices de Confusion", fontsize=16)
    plt.tight_layout()
    # Ajuster le haut pour laisser de la place au suptitle
    plt.subplots_adjust(top=0.92)
    plt.show()

if __name__ == "__main__":
    plot_confusion_matrices()
