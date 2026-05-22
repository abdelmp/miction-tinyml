import librosa
import numpy as np
import serial
import time
from pathlib import Path

AUDIO_PATH  = Path("C:/Users/asalou/S7/stages7/project/wav_audio/Audio_3.wav")
PORT        = "COM3"   # ← adapter
BAUDRATE    = 921600
SAMPLE_RATE = 16000
SEG_LEN     = 8000

# ── Charger et normaliser ─────────────────────────────────────
print("Chargement Audio_3...")
y, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)
y = y / np.max(np.abs(y))
y_int16 = (y * 32767).astype(np.int16)
print(f"Signal : {len(y_int16)} samples = {len(y_int16)/SAMPLE_RATE:.1f}s")

n_segs = len(y_int16) // SEG_LEN
print(f"Segments : {n_segs}")

# ── Ouvrir Serial SANS reset ──────────────────────────────────
ser = serial.Serial(
    PORT, BAUDRATE, timeout=5,
    dsrdtr=False, rtscts=False)
ser.setDTR(False)
ser.setRTS(False)
time.sleep(1)

# Vider le buffer
ser.reset_input_buffer()
ser.reset_output_buffer()

# ── Envoyer commande TEST ─────────────────────────────────────
print("Envoi commande TEST...")
ser.write(b"TEST\n")
time.sleep(0.5)

# Lire confirmation
resp = ser.readline().decode('utf-8', errors='ignore').strip()
print(f"ATOMS3R: {resp}")

# ── Envoyer header ────────────────────────────────────────────
ser.write(f"AUDIO:{n_segs}\n".encode())
time.sleep(0.2)

resp = ser.readline().decode('utf-8', errors='ignore').strip()
print(f"ATOMS3R: {resp}")

# ── Envoyer segments ──────────────────────────────────────────
print("Envoi audio...")
# Envoi par chunks de 256 bytes au lieu de tout d'un coup
for s in range(n_segs):
    seg = y_int16[s*SEG_LEN:(s+1)*SEG_LEN]
    data = seg.tobytes()
    
    # Envoyer par morceaux de 256 bytes
    chunk_size = 256
    for i in range(0, len(data), chunk_size):
        ser.write(data[i:i+chunk_size])
        time.sleep(0.002)  # 2ms entre chunks
    
    if s % 10 == 0:
        print(f"  Seg {s+1}/{n_segs}")
# Et augmentez l'attente finale :
print("Envoi terminé — attente résultats (5 minutes max)...")

timeout = time.time() + 300  # 5 minutes
while time.time() < timeout:
    if ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(f"ATOMS3R: {line}")
    time.sleep(0.05)

ser.close()
print("Fin")