import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_metric(configs, title, ax, metric_train, metric_val, y_label, step_y=None):
    """
    Trace les courbes (loss ou accuracy) pour une liste de configurations sur l'axe donné.
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    color_idx = 0
    
    for label, filename in configs.items():
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            
            # Lissage par moyenne mobile simple (fenêtre = 10 époques)
            smoothed_train = df[metric_train].rolling(window=10, min_periods=1).mean()
            smoothed_val = df[metric_val].rolling(window=10, min_periods=1).mean()
            
            c = colors[color_idx % len(colors)]
            
            ax.plot(df['epoch'], smoothed_train, color=c, linestyle='--', label=f'{label} (Train)')
            ax.plot(df['epoch'], smoothed_val, color=c, linestyle='-', linewidth=2, label=f'{label} (Val)')
            
            color_idx += 1
        else:
            print(f"Attention: Le fichier {filename} est introuvable.")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Époques', fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    
    # Force l'affichage des nombres sur l'axe Y (très utile quand sharey='row' les masque par défaut)
    ax.tick_params(labelleft=True)
    
    if step_y is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step_y))
        
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, linestyle=':', alpha=0.6)

def main():
    configs_dropout = {
        "Brut": "history_tactile_raw_drop.csv",
        "Dérivé": "history_tactile_deriv_drop.csv",
        "10 Canaux": "history_tactile_10ch_drop.csv"
    }

    configs_no_dropout = {
        "Brut": "history_tactile_raw.csv",
        "Dérivé": "history_tactile_deriv.csv",
        "10 Canaux": "history_tactile_10ch.csv"
    }

    # Création d'une figure avec 2 lignes (Loss et Accuracy) et 2 colonnes (Sans Dropout et Avec Dropout)
    # L'option sharey='row' force les graphiques d'une même ligne à partager la même échelle verticale
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey='row')

    # --- LIGNE 1 : LOSS ---
    plot_metric(configs_no_dropout, "Loss - Sans Dropout (300 Ép.)", axes[0, 0], 'train_loss', 'val_loss', 'Loss (Cross-Entropy)', step_y=0.2)
    plot_metric(configs_dropout, "Loss - Avec Dropout (500 Ép.)", axes[0, 1], 'train_loss', 'val_loss', 'Loss (Cross-Entropy)', step_y=0.2)

    # --- LIGNE 2 : ACCURACY ---
    plot_metric(configs_no_dropout, "Accuracy - Sans Dropout (300 Ép.)", axes[1, 0], 'train_acc', 'val_acc', 'Accuracy (%)', step_y=5)
    plot_metric(configs_dropout, "Accuracy - Avec Dropout (500 Ép.)", axes[1, 1], 'train_acc', 'val_acc', 'Accuracy (%)', step_y=5)

    plt.suptitle("Évolution des Performances (Loss et Accuracy)", fontsize=16)
    # rect=[gauche, bas, droite, haut]. On remonte le 'bas' à 0.05 pour ne pas tronquer l'axe des abscisses.
    # On passe h_pad à 3.0 pour bien espacer la ligne du haut et celle du bas.
    plt.tight_layout(rect=[0, 0.05, 1, 0.95], h_pad=3.0)
    plt.show()

if __name__ == "__main__":
    main()
