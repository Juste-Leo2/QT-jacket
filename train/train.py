import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==========================================
# 1. LE POUVOIR DE LA SEED 42
# ==========================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 2. PRÉPARATION DES DONNÉES
# ==========================================
data = np.load('mes_donnees_tactiles.npz')
X_numpy = data['X']  # Doit être [Batch, 5, 500]
y_numpy = data['y']  # Doit être [Batch] avec des entiers de 0 à 5

# 80% train / 20% validation Répartie équitablement
# Seed = 42 pour la reproductibilité
X_train, X_val, y_train, y_val = train_test_split(
    X_numpy, y_numpy, 
    test_size=0.20, 
    stratify=y_numpy, 
    random_state=42
)

# Conversion en tenseur pytorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.long)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

print(f"Entraînement : {len(X_train_t)} exemples | Validation : {len(X_val_t)} exemples")

# ==========================================
# 3. L'ARCHITECTURE : TACTILENET 
# ==========================================
class TactileNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Bloc 1 : On regarde les petites variations
        self.bloc1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Bloc 2 : On regarde les motifs moyens
        self.bloc2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Bloc 3 : On prépare la conclusion
        self.bloc3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Tête de décision
        self.classifier = nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# 4. LE MOTEUR D'ENTRAÎNEMENT
# ==========================================
model = TactileNet()
criterion = nn.CrossEntropyLoss() #Fonction Perte
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # Optimizer

epochs = 15

print("\n Lancement de l'entraînement...\n")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # --- Phase d'apprentissage ---
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad() 
        predictions = model(batch_X) # Le réseau tente de deviner
        loss = criterion(predictions, batch_y) # erereur
        loss.backward()              # backward propagation
        optimizer.step()             # escente de gradiant
        
        total_loss += loss.item()
        correct += (predictions.argmax(dim=1) == batch_y).sum().item()
        total += batch_y.size(0)
        
    train_acc = correct / total * 100
    
    # --- Phase de validation ---
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            val_correct += (predictions.argmax(dim=1) == batch_y).sum().item()
            val_total += batch_y.size(0)
            
    val_acc = val_correct / val_total * 100
    
    print(f"Époque {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Précision Train: {train_acc:.1f}% | Précision Val: {val_acc:.1f}%")

print("\nEntraînement terminé !")

# ==========================================
# 5. MATRICE DE CONFUSION SUR LA VALIDATION
# ==========================================
print("\n Génération de la matrice de confusion...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        predictions = model(batch_X)
        predicted_classes = predictions.argmax(dim=1)
        # Ajout des prédictions et vrais labels dans nos listes
        all_preds.extend(predicted_classes.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# Calcul et affichage avec scikit-learn et matplotlib
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Matrice de Confusion - Données de Validation")
plt.show()
