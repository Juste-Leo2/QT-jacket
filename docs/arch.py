import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    fig, ax = plt.subplots(figsize=(10, 13)) # Légèrement agrandi pour l'espacement
    ax.axis('off')

    # Format : (Nom, Description, Shape_out, Espace_supplementaire_en_dessous)
    steps = [
        ("Input", "Série temporelle brute", "[B, 5, 500]", 0.02),
        
        ("Bloc 1 : Conv1d + ReLU", "Filtres: 16 | Noyau: 7 | Pad: 3", "[B, 16, 500]", 0.0),
        ("Bloc 1 : MaxPool1d", "Noyau: 2 (Réduit la taille)", "[B, 16, 250]", 0.05), # Espace après Bloc 1
        
        ("Bloc 2 : Conv1d + ReLU", "Filtres: 32 | Noyau: 5 | Pad: 2", "[B, 32, 250]", 0.0),
        ("Bloc 2 : MaxPool1d", "Noyau: 2 (Réduit la taille)", "[B, 32, 125]", 0.05), # Espace après Bloc 2
        
        ("Bloc 3 : Conv1d + ReLU", "Filtres: 64 | Noyau: 3 | Pad: 1", "[B, 64, 125]", 0.0),
        ("Bloc 3 : Global Average Pooling", "Moyenne globale sur la longueur", "[B, 64, 1]", 0.05), # Espace après Bloc 3
        
        ("Flatten", "Mise à plat de la dernière dimension", "[B, 64]", 0.02),
        ("Classifier (Linear)", "Couche dense (64 -> 6 classes)", "[B, 6]", 0.0)
    ]

    box_width = 0.55
    box_height = 0.07
    x_center = 0.5  # Tout est recentré
    
    color_box = "#4C72B0"
    color_text = "white"
    color_dim = "#C44E52"

    # Calcul dynamique des positions Y pour gérer les espacements entre blocs
    y_positions = []
    current_y = 0.95
    base_gap = 0.08 # Espace de base suffisant pour la flèche et le texte

    for i in range(len(steps)):
        y_positions.append(current_y)
        if i < len(steps) - 1:
            # On soustrait la hauteur de la boîte, l'espace de base, et l'espace supplémentaire du bloc
            current_y -= (box_height + base_gap + steps[i][3])
            
    # Ajustement de la caméra
    ax.set_ylim(current_y - 0.12, 1.05)
    ax.set_xlim(0, 1)

    for i, (name, desc, shape, extra_gap) in enumerate(steps):
        y = y_positions[i]
        
        # Dessin de la boîte
        box = patches.FancyBboxPatch(
            (x_center - box_width/2, y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.02", ec="black", fc=color_box, lw=1.5
        )
        ax.add_patch(box)
        
        # Textes à l'intérieur
        ax.text(x_center, y + 0.01, name, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=color_text)
        ax.text(x_center, y - 0.015, desc, ha='center', va='center', 
                fontsize=9, color=color_text, style='italic')
        
        # Flèche et texte de la dimension
        if i < len(steps) - 1:
            next_y = y_positions[i+1]
            
            # Flèche descendant vers la prochaine boîte
            ax.annotate("", xy=(x_center, next_y + box_height/2), 
                        xytext=(x_center, y - box_height/2),
                        arrowprops=dict(arrowstyle="->", lw=2, color="black"))
            
            # Dimension positionnée EN DESSOUS de la boîte (centrée sur la flèche)
            # Le "bbox" crée un fond blanc pour "couper" la ligne de la flèche proprement
            ax.text(x_center, y - box_height/2 - 0.04, f" {shape} ", 
                    ha='center', va='center', fontsize=11, fontweight='bold', color=color_dim,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))
            
    # Prédiction finale
    last_y = y_positions[-1]
    ax.annotate("", xy=(x_center, last_y - box_height/2 - 0.06), 
                xytext=(x_center, last_y - box_height/2),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.text(x_center, last_y - box_height/2 - 0.09, "Prédictions finales [B, 6]", 
            ha='center', va='center', fontsize=13, fontweight='bold', color="#55A868")

    plt.title("Architecture TactileNet : Blocs et Dimensions", fontsize=16, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.show()

# Exécuter la fonction
draw_architecture()
