import torch
import torch.nn as nn
import os
import argparse

# ==========================================
# 1. PARSING DES ARGUMENTS LIGNE DE COMMANDE
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Export de TactileNet en ONNX")
    parser.add_argument("--dropout", action="store_true", help="Le modèle a été entraîné avec dropout")
    parser.add_argument("--derivate", action="store_true", help="Le modèle utilise les signaux dérivés")
    parser.add_argument("--extend", action="store_true", help="Le modèle utilise 10 canaux")
    return parser.parse_args()

# ==========================================
# 2. ARCHITECTURE DU MODÈLE (TACTILENET)
# ==========================================
class TactileNet(nn.Module):
    def __init__(self, in_channels=5, use_dropout=False):
        super().__init__()
        self.use_dropout = use_dropout
        
        self.input_norm = nn.BatchNorm1d(in_channels)
        
        self.bloc1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.bloc2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.bloc3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)
            
        self.classifier = nn.Linear(in_features=64, out_features=5)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        x = torch.flatten(x, 1)
        
        if self.use_dropout:
            x = self.dropout(x)
            
        x = self.classifier(x)
        return x

def main():
    args = parse_args()
    
    num_channels = 10 if args.extend else 5
    
    # Reconstruction du nom du modèle en fonction des arguments
    nom_exp = "tactile"
    if args.extend:
        nom_exp += "_10ch"
    elif args.derivate:
        nom_exp += "_deriv"
    else:
        nom_exp += "_raw"
        
    if args.dropout:
        nom_exp += "_drop"
        
    pth_path = f"{nom_exp}_model.pth"
    onnx_path = f"{nom_exp}_model.onnx"

    print(f"Chargement des poids depuis : {pth_path}")
    if not os.path.exists(pth_path):
        print("Erreur : Le fichier .pth n'a pas été trouvé. Avez-vous terminé l'entraînement avec ces options ?")
        return

    # 3. Initialisation du modèle en mode évaluation
    model = TactileNet(in_channels=num_channels, use_dropout=args.dropout)
    model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    
    # /!\ TRÈS IMPORTANT : model.eval() fige la BatchNorm et le Dropout
    model.eval()  

    # 4. Création d'une entrée "fantôme" (dummy input)
    # Le format attendu par le modèle est : [Batch_Size, Nombre_de_Canaux, Longueur_Temporelle]
    dummy_input = torch.randn(1, num_channels, 500)

    # 5. Exportation ONNX
    print(f"Exportation du graphe ONNX vers : {onnx_path}")
    torch.onnx.export(
        model,                      # Le modèle PyTorch
        dummy_input,                # L'entrée fantôme pour tracer le graphe
        onnx_path,                  # Le chemin de sauvegarde
        export_params=True,         # Sauvegarder les poids à l'intérieur
        opset_version=11,           # Version standard stable de ONNX
        do_constant_folding=True,   # Optimisation : calcule en avance ce qui ne change pas
        input_names=['input'],      # On nomme l'entrée
        output_names=['output'],    # On nomme la sortie
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("\nSuccès ! Votre modèle ONNX est prêt à être utilisé par onnxruntime.")

if __name__ == "__main__":
    main()
