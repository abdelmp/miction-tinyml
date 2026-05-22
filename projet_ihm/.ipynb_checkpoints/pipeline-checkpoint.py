# ════════════════════════════════════════════════════════════════
# pipeline.py 
# ════════════════════════════════════════════════════════════════
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from collections import Counter

TARGET_SR    = 16000
SEGMENT_MS   = 500
HOP_MS       = 250
SEG_LEN      = int(SEGMENT_MS * TARGET_SR / 1000)   # 8000
HOP_LEN_AUD  = int(HOP_MS     * TARGET_SR / 1000)   # 4000
N_MELS       = 64
N_FFT        = 512
HOP_LEN_MEL  = 160
FMAX         = 6000
SEUIL_HACHE  = 2
SEUIL_FAIBLE = 8.0
CLASSES = ['bruit_ambiant', 'bruit_chasse', 'miction_active']

def charger_modele(path_keras):
    model = tf.keras.models.load_model(path_keras)
    le    = LabelEncoder()
    le.fit(CLASSES)
    print(f" Modèle chargé — classes : {le.classes_}")
    return model, le

def charger_audio(chemin):
    y, sr = librosa.load(chemin, sr=TARGET_SR, mono=True)
    return y.astype(np.float32), sr

def signal_to_melspec(segment):

    mel    = librosa.feature.melspectrogram(
        y=segment.astype(float), sr=TARGET_SR,
        n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LEN_MEL, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mn, mx = mel_db.min(), mel_db.max()
    if mx - mn > 0:
        mel_norm = (mel_db - mn) / (mx - mn)
    else:
        mel_norm = np.zeros_like(mel_db)
    return mel_norm.astype(np.float32)

def corriger_contexte(labels, fenetre=1):
    labels = list(labels)
    for i in range(fenetre, len(labels) - fenetre):
        if labels[i-fenetre] == labels[i+fenetre] != labels[i]:
            labels[i] = labels[i-fenetre]
    return labels

def corriger_pipeline_complet(labels_pred, fenetre=1, gap=4):
    labels = corriger_contexte(labels_pred, fenetre)

    # Passe 2 — fusion épisodes miction (trous 1-3 segments)
    for _ in range(gap * 3):
        changed = False
        for trou in range(1, 4):
            for i in range(len(labels) - trou - 1):
                if (labels[i] == 'miction_active' and
                    labels[i + trou + 1] == 'miction_active' and
                    all(labels[i+k] != 'miction_active'
                        for k in range(1, trou + 1))):
                    for k in range(1, trou + 1):
                        labels[i + k] = 'miction_active'
                    changed = True
        if not changed:
            break

    # Passe 3 — supprimer TOUS les segments isolés
    # (pas seulement miction_active — tous les labels)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(labels) - 1):
            # Regarder fenêtre ±2
            contexte = []
            for k in range(max(0,i-2), min(len(labels),i+3)):
                if k != i:
                    contexte.append(labels[k])

            # Si le segment est différent de tous ses voisins proches
            vote = max(set(contexte), key=contexte.count)
            if contexte.count(labels[i]) <= 1 and vote != labels[i]:
                labels[i] = vote
                changed = True

    return labels
    
def classifier_type(n_episodes, duree_s):
    if duree_s < SEUIL_FAIBLE:      return 'jet_faible'
    elif n_episodes >= SEUIL_HACHE: return 'jet_hache'
    else:                            return 'jet_continu'

def calculer_indicateurs(labels_pred):
    counts    = Counter(labels_pred)
    n_miction = counts.get('miction_active', 0)
    duree_s   = round(n_miction * HOP_MS / 1000, 2)
    chasse    = counts.get('bruit_chasse', 0) > 0

    episodes, en_miction, debut = [], False, 0
    for i, lbl in enumerate(labels_pred):
        if lbl == 'miction_active' and not en_miction:
            en_miction, debut = True, i
        elif lbl != 'miction_active' and en_miction:
            en_miction = False
            episodes.append({
                'debut': round(debut * HOP_MS / 1000, 2),
                'fin':   round(i     * HOP_MS / 1000, 2),
                'duree': round((i - debut) * HOP_MS / 1000, 2)
            })
    if en_miction:
        n = len(labels_pred)
        episodes.append({
            'debut': round(debut * HOP_MS / 1000, 2),
            'fin':   round(n     * HOP_MS / 1000, 2),
            'duree': round((n - debut) * HOP_MS / 1000, 2)
        })

    return {
        'duree_miction_s': duree_s,
        'nb_episodes':     len(episodes),
        'type_miction':    classifier_type(len(episodes), duree_s),
        'chasse_detectee': chasse,
        'episodes':        episodes,
        'distribution':    dict(counts),
        'n_segments':      len(labels_pred),
    }

def analyser_audio(chemin_audio, model, le):
    """Pipeline complet identique à l'entraînement."""

    # 1. Charger avec librosa — IDENTIQUE à dataset['filtered']
    y, sr = charger_audio(chemin_audio)
    print(f"Signal : {len(y)} samples = {len(y)/sr:.1f}s  "
          f"max={np.max(np.abs(y)):.4f}")

    # 2. Segmenter avec chevauchement 50%
    segments, t_starts = [], []
    pos = 0
    while pos + SEG_LEN <= len(y):
        segments.append(y[pos:pos + SEG_LEN].copy())
        t_starts.append(round(pos / sr, 3))
        pos += HOP_LEN_AUD

    if not segments:
        return None

    print(f"Segments : {len(segments)}")

    # 3. Spectrogrammes Mel
    X_mel = np.array([signal_to_melspec(s) for s in segments],
                      dtype=np.float32)[..., np.newaxis]
    print(f"X_mel : {X_mel.shape}")

 
    # 4. Prédiction
    probas      = model.predict(X_mel, verbose=0)
    confidences = np.max(probas, axis=1).tolist()
    y_pred      = np.argmax(probas, axis=1)

    # ← AJOUTER : clip les indices à 0-2 (sécurité)
    y_pred = np.clip(y_pred, 0, len(CLASSES) - 1)

    labels_pred = list(le.inverse_transform(y_pred))
    # Fusion silence → ambiant (sécurité ancien modèle)
    labels_pred = ['bruit_ambiant' if l == 'silence'
                   else l for l in labels_pred]


    # 5. Correction
    labels_corr = corriger_pipeline_complet(labels_pred, fenetre=1, gap=4)

    # 6. Indicateurs
    indic = calculer_indicateurs(labels_corr)
    indic['labels_pred']  = labels_corr
    indic['confidences']  = confidences
    indic['t_starts']     = t_starts
    indic['duree_totale'] = round(len(y) / sr, 2)

    return indic