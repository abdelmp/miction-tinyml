# receive_and_plot.py — lancer sur PC
import serial
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

PORT    = "COM3"   # ← adapter
BAUD    = 921600
SR      = 16000

print(f"Connexion sur {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=30)

samples_brut   = []
samples_norm   = []
n_samples      = 0
mode           = None

print("En attente des données ATOMS3R...")
print("Appuyez sur le bouton de l'ATOMS3R pour envoyer")

while True:
    line = ser.readline().decode(errors='ignore').strip()
    if not line:
        continue

    if line == "=== DEBUT_AUDIO ===":
        print("Reception en cours...")

    elif line.startswith("SAMPLES:"):
        n_samples = int(line.split(":")[1])
        print(f"  → {n_samples} samples ({n_samples/SR:.1f}s)")

    elif line == "=== BRUT ===":
        mode = "brut"
        print("  → Signal brut...")

    elif line == "=== NORMALISE ===":
        mode = "norm"
        print("  → Signal normalise...")
        print(f"  → Brut recu: {len(samples_brut)} samples")

    elif line == "=== FIN_AUDIO ===":
        print(f"  → Normalise recu: {len(samples_norm)} samples")
        break

    else:
        try:
            v = int(line)
            if mode == "brut":
                samples_brut.append(v)
            elif mode == "norm":
                samples_norm.append(v)
        except:
            pass

ser.close()

# Convertir en float
brut = np.array(samples_brut, dtype=np.int16)
norm = np.array(samples_norm, dtype=np.int16)
brut_f = brut.astype(np.float32) / 32768.0
norm_f = norm.astype(np.float32) / 32768.0

print(f"\nStats signal brut      : max={np.max(np.abs(brut_f)):.4f}  moy={np.mean(np.abs(brut_f)):.4f}")
print(f"Stats signal normalise : max={np.max(np.abs(norm_f)):.4f}  moy={np.mean(np.abs(norm_f)):.4f}")

# Sauvegarder WAV
sf.write("atoms3r_brut.wav",  brut_f, SR)
sf.write("atoms3r_norm.wav",  norm_f, SR)
print("\nFichiers sauvegardes : atoms3r_brut.wav  atoms3r_norm.wav")

# Afficher
t = np.linspace(0, len(norm_f)/SR, len(norm_f))
fig, axes = plt.subplots(2, 1, figsize=(14, 6))
fig.suptitle("Comparaison signal ATOMS3R", fontweight='bold')

axes[0].plot(t, brut_f, color='orange', lw=0.5)
axes[0].set_title(f"Signal brut INMP441  (max={np.max(np.abs(brut_f)):.4f})")
axes[0].set_ylabel("Amplitude"); axes[0].set_xlabel("Temps (s)")

axes[1].plot(t, norm_f, color='blue', lw=0.5)
axes[1].set_title(f"Signal normalisé  (max={np.max(np.abs(norm_f)):.4f})")
axes[1].set_ylabel("Amplitude"); axes[1].set_xlabel("Temps (s)")

plt.tight_layout()
plt.savefig("comparaison_signal.png", dpi=150)
plt.show()