import torch
import torch.nn as nn
import os

# ==========================================
# 1. ARCHITECTURE DU MODÈLE (TACTILENET)
# ==========================================
# On doit redéfinir la structure exacte du réseau pour pouvoir charger les poids
class TactileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bloc1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=16, kernel_size=7, padding=3),
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
        self.classifier = nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    pth_path = "tactile_model.pth"
    onnx_path = "veste_model.onnx"

    print(f"Chargement des poids depuis : {pth_path}")
    if not os.path.exists(pth_path):
        print("Erreur : Le fichier .pth n'a pas été trouvé. Avez-vous terminé l'entraînement ?")
        return

    # 2. Initialisation du modèle en mode évaluation
    model = TactileNet()
    model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    model.eval()  # Très important pour figer le réseau (pas de gradients)

    # 3. Création d'une entrée "fantôme" (dummy input)
    # Le format attendu par le modèle est : [Batch_Size, Nombre_de_Canaux, Longueur_Temporelle]
    # Soit : [1, 5, 500] (1 exemple, 5 capteurs, 500 points)
    dummy_input = torch.randn(1, 5, 500)

    # 4. Exportation ONNX
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

    print("\nSuccès ! Votre modèle .onnx est prêt à être utilisé par onnxruntime.")

if __name__ == "__main__":
    main()
