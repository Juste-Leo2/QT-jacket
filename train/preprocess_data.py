import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def compute_derivative(arr):
    """
    Compute classical numerical derivative.
    Forward difference for the first point, backward difference for the last point,
    and central difference for the interior points.
    arr has shape (N, C) where N is number of points and C is number of channels.
    """
    deriv = np.zeros_like(arr)
    if arr.shape[0] < 3:
        # Fallback if the array is too small
        if arr.shape[0] >= 2:
            deriv[0, :] = arr[1, :] - arr[0, :]
            deriv[1, :] = arr[1, :] - arr[0, :]
        return deriv
        
    # Forward difference for the first point
    deriv[0, :] = arr[1, :] - arr[0, :]
    
    # Backward difference for the last point
    deriv[-1, :] = arr[-1, :] - arr[-2, :]
    
    # Central difference for the rest
    deriv[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / 2.0
    
    return deriv

CHANNEL_NAMES = ["Bras G", "Bras D", "Torse G", "Torse D", "Dos"]

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data_collected")
    
    npz_files = glob.glob(os.path.join(data_dir, "class_*.npz"))
    # Filter out already processed files if any
    npz_files = [f for f in npz_files if "100Hz" not in f and "deriv" not in f]
    
    if not npz_files:
        print("Aucun fichier .npz source trouvé.")
        return
        
    plotted = False
    
    print(f"Trouvé {len(npz_files)} fichiers .npz à traiter.")
    
    for filepath in npz_files:
        filename = os.path.basename(filepath)
        # Extract class id (e.g., "class_0.npz" -> "0")
        class_id = filename.replace("class_", "").replace(".npz", "")
        
        data = np.load(filepath)
        
        dict_100Hz = {}
        dict_deriv = {}
        
        for key in data.files:
            sample_1kHz = data[key]
            
            # 1. Downsample (Moyenne sur 10 points)
            # e.g., 5000 points -> 500 points
            N, C = sample_1kHz.shape
            
            # Ensure N is divisible by 10 for simplicity
            if N % 10 != 0:
                truncate_len = N - (N % 10)
                sample_1kHz = sample_1kHz[:truncate_len, :]
                N = truncate_len
                
            # Reshape and compute mean on axis 1 (the blocks of 10)
            sample_100Hz = sample_1kHz.reshape(N // 10, 10, C).mean(axis=1)
            
            # 2. Derivative
            sample_deriv = compute_derivative(sample_100Hz)
            
            dict_100Hz[key] = sample_100Hz
            dict_deriv[key] = sample_deriv
            
            # 3. Plot for the very first sample of class 3 encountered
            if not plotted and class_id == "3":
                fig, axes = plt.subplots(3, 1, figsize=(12, 10))
                
                # Plot 1kHz
                time_1kHz = np.linspace(0, N/1000, N)
                for c in range(C):
                    axes[0].plot(time_1kHz, sample_1kHz[:, c], label=CHANNEL_NAMES[c])
                axes[0].set_title("Données originales (1kHz)")
                axes[0].set_xlabel("Temps (s)")
                axes[0].set_ylabel("Valeur")
                axes[0].legend(loc='upper right')
                
                # Plot 100Hz
                time_100Hz = np.linspace(0, N/1000, N // 10)
                for c in range(C):
                    axes[1].plot(time_100Hz, sample_100Hz[:, c], label=CHANNEL_NAMES[c])
                axes[1].set_title("Données moyennées (100Hz)")
                axes[1].set_xlabel("Temps (s)")
                axes[1].set_ylabel("Valeur")
                axes[1].legend(loc='upper right')
                
                # Plot Derivative
                for c in range(C):
                    axes[2].plot(time_100Hz, sample_deriv[:, c], label=CHANNEL_NAMES[c])
                axes[2].set_title("Dérivée numérique des données à 100Hz")
                axes[2].set_xlabel("Temps (s)")
                axes[2].set_ylabel("Dérivée")
                axes[2].legend(loc='upper right')
                
                plt.tight_layout(h_pad=2.0)
                plt.show()
                plotted = True
                
        # Save new files
        path_100Hz = os.path.join(data_dir, f"class_{class_id}_100Hz.npz")
        path_deriv = os.path.join(data_dir, f"class_{class_id}_100Hz_deriv.npz")
        
        np.savez(path_100Hz, **dict_100Hz)
        np.savez(path_deriv, **dict_deriv)
        
        print(f"Sauvegardé {path_100Hz} et {path_deriv}")

    print("Traitement terminé.")

if __name__ == '__main__':
    main()
