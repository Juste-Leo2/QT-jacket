import argparse
import os
import glob
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# matplotlib et ConfusionMatrixDisplay retirés car on fera les tracés ailleurs

# ==========================================
# 1. PARSING DES ARGUMENTS LIGNE DE COMMANDE
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement de TactileNet")
    parser.add_argument("--dropout", action="store_true", help="Appliquer un dropout de 0.5 avant la classification")
    parser.add_argument("--derivate", action="store_true", help="Utiliser les signaux dérivés (100Hz_deriv)")
    parser.add_argument("--extend", action="store_true", help="Utiliser 10 canaux (combine signaux bruts et dérivés)")
    return parser.parse_args()

args = parse_args()

# ==========================================
# 2. LE POUVOIR DE LA SEED 42
# ==========================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 3. PRÉPARATION DES DONNÉES
# ==========================================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data_collected")

if args.extend:
    # Recherche des fichiers de base (bruts), on chargera les dérivés dynamiquement
    npz_files_base = glob.glob(os.path.join(data_dir, "class_*_100Hz.npz"))
elif args.derivate:
    # Recherche uniquement des dérivés
    npz_files_base = glob.glob(os.path.join(data_dir, "class_*_100Hz_deriv.npz"))
else:
    # Recherche uniquement des bruts (comportement par défaut)
    npz_files_base = glob.glob(os.path.join(data_dir, "class_*_100Hz.npz"))

X_list = []
y_list = []

for filepath in npz_files_base:
    filename = os.path.basename(filepath)
    class_id = int(filename.split('_')[1])
    
    # On ignore la classe 5 par sécurité, même si on est censé ne plus l'avoir acquise
    if class_id == 5:
        continue
    
    if args.extend:
        # Combiner raw et deriv
        filepath_deriv = filepath.replace("_100Hz.npz", "_100Hz_deriv.npz")
        if not os.path.exists(filepath_deriv):
            print(f"Attention, fichier dérivé manquant pour {filename}. On saute cet échantillon.")
            continue
            
        data_raw = np.load(filepath)
        data_deriv = np.load(filepath_deriv)
        
        for key in data_raw.files:
            if key in data_deriv.files:
                sample_raw = data_raw[key]      # shape [500, 5]
                sample_deriv = data_deriv[key]  # shape [500, 5]
                
                # Concaténation le long de l'axe des canaux
                sample_combined = np.concatenate([sample_raw, sample_deriv], axis=1) # [500, 10]
                
                # Transpose to [10, 500] for PyTorch Conv1d
                sample_combined = sample_combined.transpose(1, 0)
                X_list.append(sample_combined)
                y_list.append(class_id)
    else:
        # Standard: charger juste le fichier ciblé
        data = np.load(filepath)
        for key in data.files:
            sample = data[key]  # shape [500, 5]
            # Transpose to [5, 500] for PyTorch Conv1d
            sample = sample.transpose(1, 0)
            X_list.append(sample)
            y_list.append(class_id)

if len(X_list) == 0:
    print("Erreur: Aucune donnée trouvée. Vérifiez que les fichiers NPZ sont présents dans data_collected.")
    exit(1)

X_numpy = np.array(X_list)  # [Batch, Channels, 500]
y_numpy = np.array(y_list)  # [Batch]

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
print(f"Shape des données d'entrée : {X_train_t.shape}")

# ==========================================
# 4. L'ARCHITECTURE : TACTILENET 
# ==========================================
class TactileNet(nn.Module):
    def __init__(self, in_channels=5, use_dropout=False):
        super().__init__()
        
        self.use_dropout = use_dropout
        
        # Normalisation des données en entrée
        self.input_norm = nn.BatchNorm1d(num_features=in_channels)
        
        # Bloc 1 : On regarde les petites variations
        self.bloc1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=7, padding=3),
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
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)
            
        self.classifier = nn.Linear(in_features=64, out_features=5)

    def forward(self, x):
        # Normalisation des données dès l'entrée du réseau
        x = self.input_norm(x)
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        x = torch.flatten(x, 1)
        
        if self.use_dropout:
            x = self.dropout(x)
            
        x = self.classifier(x)
        return x

# ==========================================
# 5. LE MOTEUR D'ENTRAÎNEMENT
# ==========================================
num_channels = 10 if args.extend else 5
model = TactileNet(in_channels=num_channels, use_dropout=args.dropout)

criterion = nn.CrossEntropyLoss() #Fonction Perte
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # Optimizer

# 500 époques avec dropout, 300 sans dropout
epochs = 500 if args.dropout else 300

print(f"\n Lancement de l'entraînement avec {num_channels} canaux (Dropout: {args.dropout}, Epochs: {epochs})...\n")

history = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

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
    val_loss_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            val_loss_total += loss.item()
            
            val_correct += (predictions.argmax(dim=1) == batch_y).sum().item()
            val_total += batch_y.size(0)
            
    val_acc = val_correct / val_total * 100
    val_loss_avg = val_loss_total / len(val_loader)
    train_loss_avg = total_loss / len(train_loader)
    
    # Enregistrement dans l'historique
    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss_avg)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss_avg)
    history["val_acc"].append(val_acc)
    
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        print(f"Époque {epoch+1:03d}/{epochs} | Loss Train: {train_loss_avg:.4f} | Loss Val: {val_loss_avg:.4f} | Précision Train: {train_acc:.1f}% | Précision Val: {val_acc:.1f}%")

print("\nEntraînement terminé !")

# ==========================================
# 6. SAUVEGARDE DES RESULTATS (EVAL)
# ==========================================
# Dossier d'évaluation
eval_dir = os.path.join(base_dir, "eval")
os.makedirs(eval_dir, exist_ok=True)

# Nom de l'expérience en fonction des paramètres
nom_exp = "tactile"
if args.extend:
    nom_exp += "_10ch"
elif args.derivate:
    nom_exp += "_deriv"
else:
    nom_exp += "_raw"
    
if args.dropout:
    nom_exp += "_drop"

# 6.1 Sauvegarde de l'historique CSV
csv_path = os.path.join(eval_dir, f"history_{nom_exp}.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    for i in range(len(history["epoch"])):
        writer.writerow([
            history["epoch"][i],
            history["train_loss"][i],
            history["train_acc"][i],
            history["val_loss"][i],
            history["val_acc"][i]
        ])
print(f"Historique sauvegardé sous : {csv_path}")

# 6.2 Calcul et sauvegarde de la matrice de confusion
print("\n Génération de la matrice de confusion...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        predictions = model(batch_X)
        predicted_classes = predictions.argmax(dim=1)
        all_preds.extend(predicted_classes.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
cm_path = os.path.join(eval_dir, f"cm_{nom_exp}.npy")
np.save(cm_path, cm)
print(f"Matrice de confusion sauvegardée sous : {cm_path}")

# ==========================================
# 7. SAUVEGARDE DU MODÈLE
# ==========================================
model_path = os.path.join(base_dir, f"{nom_exp}_model.pth")
torch.save(model.state_dict(), model_path)
print(f"\nModèle sauvegardé sous : {model_path}")
