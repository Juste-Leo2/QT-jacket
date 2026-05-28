Voici ton README mis à jour, plus direct, sans la méthode X11, sans émojis, et adapté à tes paramètres réseau exacts :

---

# Tutoriel : Afficher l'interface graphique du Raspberry Pi sur un PC Linux

Étant donné que ton ordinateur principal est sous Linux, tu as un énorme avantage : Linux gère nativement l'affichage déporté. Tu n'as pas besoin d'installer de logiciels tiers lourds.

Voici la méthode la plus fiable pour afficher l'interface graphique de `acquisition.py` sur ton PC via le réseau local.

---

## Méthode : VNC via TigerVNC (La plus fluide et fiable)

VNC compresse l'image du bureau complet du Raspberry Pi et te l'envoie. Le serveur VNC par défaut du Raspberry Pi utilise un chiffrement spécifique qui cause souvent des erreurs avec les clients standards. Il est donc fortement recommandé d'utiliser **TigerVNC**.

### 1. Activer VNC sur le Raspberry Pi

Connecte-toi en SSH sur ton Pi et lance l'outil de configuration :

```bash
ssh qt@192.168.100.3
sudo raspi-config

```

* Va dans **Interface Options** (ou *Interfacing Options*).
* Sélectionne **VNC** et choisis **Yes / Oui** pour l'activer.
* Quitte l'outil (`Finish`).

### 2. Installer TigerVNC sur ton PC Linux

Sur ton PC Linux, ouvre un terminal et installe le client VNC :

```bash
# Sur Ubuntu / Debian / Linux Mint :
sudo apt install tigervnc-viewer

# Sur Arch Linux / Manjaro :
sudo pacman -S tigervnc

```

### 3. Se connecter

* Ouvre l'application **TigerVNC Viewer** sur ton PC.
* Dans la petite fenêtre qui apparaît, entre l'adresse IP de ton Raspberry Pi : **`192.168.100.3`**.
* Clique sur **Connect**.
* Entre ton mot de passe (ex: `qtrobot`) pour valider.

> **Résultat :** Tu auras le bureau complet du Raspberry Pi affiché dans une fenêtre fluide. Ouvre un terminal dans ce bureau et lance ton script `acquisition.py` de manière tout à fait classique !

---

## Astuce : Résoudre l'accès à Internet

Si ton Raspberry Pi est branché directement à ton PC avec un câble Ethernet (sur l'IP `192.168.100.3`) tout en étant connecté au Wi-Fi, et que **tu n'arrives pas à utiliser Internet**, c'est à cause d'un conflit de priorité réseau.

Pour rediriger Internet vers le Wi-Fi sans couper ta connexion VNC locale, ouvre le terminal du Raspberry Pi (depuis ton bureau VNC) et tape cette commande :

```bash
sudo ip route del default via 192.168.100.1 dev eth0

```

Cette solution est temporaire. Au prochain redémarrage du Raspberry Pi, le comportement par défaut sera restauré.

Retour au [tuto](./tuto.md)