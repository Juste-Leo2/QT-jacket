# Tutoriel : Afficher l'interface graphique du Raspberry Pi sur un PC Linux

Étant donné que ton ordinateur principal est sous Linux, tu as un énorme avantage : Linux gère nativement l'affichage déporté (X11) ! Tu n'as pas besoin d'installer de logiciels tiers lourds comme sur Windows.

Voici les deux meilleures méthodes pour afficher l'interface graphique de `acquisition.py` sur ton PC.

---

## Méthode 1 : Le "X11 Forwarding" via SSH (La plus rapide à tester)

Le *X11 Forwarding* permet de faire passer la fenêtre de l'application à travers ta connexion SSH. L'interface s'ouvrira sur ton écran Linux comme si c'était une application locale.

### Prérequis (sur le Raspberry Pi) :
Il faut s'assurer que le Raspberry Pi autorise le transfert d'affichage.
1. Connecte-toi normalement en SSH : `ssh pi@adresse_ip_du_pi`
2. Ouvre le fichier de configuration SSH (optionnel, souvent activé par défaut) :
   ```bash
   sudo nano /etc/ssh/sshd_config
   ```
3. Cherche la ligne `X11Forwarding` et assure-toi qu'elle est sur `yes` :
   ```text
   X11Forwarding yes
   ```
4. Redémarre le service SSH si tu as fait une modification : `sudo systemctl restart ssh`

### Comment l'utiliser :
Depuis le terminal de ton **PC Linux**, au lieu d'utiliser `ssh` basique, ajoute l'option `-X` (ou `-Y` pour passer outre certaines sécurités locales, souvent recommandé) :

```bash
ssh -X pi@adresse_ip_du_pi
# ou
ssh -Y pi@adresse_ip_du_pi
```

Une fois connecté, navigue dans ton dossier et lance le script :
```bash
cd /chemin/vers/QT-jacket
python acquisition.py
```
> **Résultat :** La fenêtre apparaîtra sur ton PC Linux. 
> *Note : Comme l'interface met à jour des graphiques et des couleurs en temps réel, l'affichage via SSH `-X` peut parfois paraître légèrement saccadé si la connexion Wi-Fi n'est pas parfaite.*

---

## Méthode 2 : VNC (Pour plus de fluidité)

Si le *X11 Forwarding* est trop lent ou saccadé à cause du réseau, VNC est l'alternative idéale. VNC compresse l'image du bureau complet du Raspberry Pi et te l'envoie.

### 1. Activer VNC sur le Raspberry Pi
Connecte-toi en SSH sur ton Pi et lance l'outil de configuration :
```bash
sudo raspi-config
```
- Va dans **Interface Options** (ou *Interfacing Options*).
- Sélectionne **VNC** et choisis **Yes / Oui** pour l'activer.
- Quitte l'outil (`Finish`).

### 2. Installer un client VNC sur ton PC Linux
Sur ton PC Linux, ouvre un terminal et installe un client VNC (par exemple `Remmina` ou `TigerVNC`, très courants sur Ubuntu/Debian) :

```bash
# Sur Ubuntu / Debian :
sudo apt install remmina

# Sur Arch Linux :
sudo pacman -S remmina
```

### 3. Se connecter
- Ouvre **Remmina** (ou ton client VNC) sur ton PC.
- Crée une nouvelle connexion de type **VNC**.
- Entre l'adresse IP de ton Raspberry Pi.
- Connecte-toi avec tes identifiants (utilisateur `pi`, et ton mot de passe).

> **Résultat :** Tu auras le bureau complet du Raspberry Pi affiché dans une fenêtre fluide. Ouvre un terminal dans ce bureau et lance ton script `acquisition.py` de manière tout à fait classique !
