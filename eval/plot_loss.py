import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric(configs, title, ax, metric_train, metric_val, y_label):
    """
    Trace les courbes (loss ou accuracy) pour une liste de configurations sur l'axe donné.
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    color_idx = 0
    
    for label, filename in configs.items():
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            
            c = colors[color_idx % len(colors)]
            
            ax.plot(df['epoch'], df[metric_train], color=c, linestyle='--', label=f'{label} (Train)')
            ax.plot(df['epoch'], df[metric_val], color=c, linestyle='-', linewidth=2, label=f'{label} (Val)')
            
            color_idx += 1
        else:
            print(f"Attention: Le fichier {filename} est introuvable.")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Époques', fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, linestyle=':', alpha=0.6)

def main():
    configs_dropout = {
        "Raw": "history_tactile_raw_drop.csv",
        "Deriv": "history_tactile_deriv_drop.csv",
        "10ch": "history_tactile_10ch_drop.csv"
    }

    configs_no_dropout = {
        "Raw": "history_tactile_raw.csv",
        "Deriv": "history_tactile_deriv.csv",
        "10ch": "history_tactile_10ch.csv"
    }

    # Création d'une figure avec 2 lignes (Loss et Accuracy) et 2 colonnes (Sans Dropout et Avec Dropout)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # --- LIGNE 1 : LOSS ---
    plot_metric(configs_no_dropout, "Loss - Sans Dropout (300 Ép.)", axes[0, 0], 'train_loss', 'val_loss', 'Loss (Cross-Entropy)')
    plot_metric(configs_dropout, "Loss - Avec Dropout (500 Ép.)", axes[0, 1], 'train_loss', 'val_loss', 'Loss (Cross-Entropy)')

    # --- LIGNE 2 : ACCURACY ---
    plot_metric(configs_no_dropout, "Accuracy - Sans Dropout (300 Ép.)", axes[1, 0], 'train_acc', 'val_acc', 'Accuracy (%)')
    plot_metric(configs_dropout, "Accuracy - Avec Dropout (500 Ép.)", axes[1, 1], 'train_acc', 'val_acc', 'Accuracy (%)')

    plt.suptitle("Évolution des Performances (Loss et Accuracy)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

if __name__ == "__main__":
    main()
