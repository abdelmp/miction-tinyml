# ════════════════════════════════════════════════════════════════
# app.py — Serveur Flask
# ════════════════════════════════════════════════════════════════
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os, json, time, threading
from pipeline import charger_modele, analyser_audio
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from db import init_db, sauvegarder_bilan, get_historique, get_stats, get_activite, supprimer_bilan

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = Path("C:/Users/asalou/S7/stages7/project/augmentation/aug/cnn2d_v2.keras")
if not MODEL_PATH.exists():
    print(f"ALERTE : Le fichier {MODEL_PATH} est introuvable !")
else:
    model = tf.keras.models.load_model(MODEL_PATH)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_db()

print("Chargement du modèle CNN...")
model, le = charger_modele(MODEL_PATH)
print("Modèle chargé")

simulation_state = {
    'running':    False,
    'fichier':    None,
    'resultats':  [],
    'progress':   0,
}

# ── Bilans ATOMS3R ─────────────────────────────────────────────
bilans_recus = []

# ── État temps réel ATOMS3R ───────────────────────────────────
atoms3r_live = {
    'active':        False,
    'seg_courant':   0,
    'total_segs':    0,
    'label_courant': 3,
    'duree_miction': 0.0,
    'type_miction':  '---',
    'labels':        [],
}

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier'}), 400
    f      = request.files['file']
    chemin = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(chemin)
    try:
        t0    = time.time()
        indic = analyser_audio(chemin, model, le)
        indic['temps_analyse'] = round(time.time() - t0, 2)
        return jsonify({'success': True, 'resultats': indic})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate/start', methods=['POST'])
def simulate_start():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier'}), 400
    f      = request.files['file']
    chemin = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(chemin)
    simulation_state['running']   = True
    simulation_state['resultats'] = []
    simulation_state['progress']  = 0
    simulation_state['fichier']   = chemin
    return jsonify({'success': True})

@app.route('/api/simulate/status')
def simulate_status():
    return jsonify(simulation_state)

@app.route('/api/simulate/stop')
def simulate_stop():
    simulation_state['running'] = False
    return jsonify({'success': True})

# ── Routes ATOMS3R ────────────────────────────────────────────
@app.route('/api/atoms3r/bilan', methods=['POST'])
def atoms3r_bilan():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON invalide'}), 400
        data['timestamp'] = datetime.now().strftime('%H:%M:%S')
        data['source']    = 'ATOMS3R'
        bilans_recus.append(data)
        atoms3r_live['active'] = False

        sauvegarder_bilan(data)   # ← AJOUTER CETTE LIGNE

        print(f"\n Bilan ATOMS3R — {data['duree_miction']}s {data['type_miction']}")
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/atoms3r/last', methods=['GET'])
def atoms3r_last():
    try:
        # Lire depuis SQLite — persiste même après redémarrage
        bilans = get_historique(limit=1)
        if bilans:
            b = bilans[0]
            # Reformater pour compatibilité avec l'interface
            b['source']    = 'ATOMS3R'
            b['timestamp'] = b['heure']
            return jsonify({'success': True, 'bilan': b})
        return jsonify({'success': False, 'message': 'Aucun bilan'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
@app.route('/api/atoms3r/update', methods=['POST'])
def atoms3r_update():
    try:
        data = request.get_json()
        atoms3r_live['active']        = True
        atoms3r_live['seg_courant']   = data.get('seg', 0)
        atoms3r_live['total_segs']    = data.get('total_segs', 0)
        atoms3r_live['label_courant'] = data.get('label', 3)
        atoms3r_live['duree_miction'] = data.get('duree_miction', 0)
        atoms3r_live['type_miction']  = data.get('type_miction', '---')
        atoms3r_live['labels'].append(data.get('label', 3))
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/atoms3r/live', methods=['GET'])
def atoms3r_live_status():
    return jsonify({'success': True, 'live': atoms3r_live})

@app.route('/api/atoms3r/reset', methods=['POST'])
def atoms3r_reset():
    atoms3r_live['active']        = False
    atoms3r_live['seg_courant']   = 0
    atoms3r_live['total_segs']    = 0
    atoms3r_live['label_courant'] = 3
    atoms3r_live['duree_miction'] = 0.0
    atoms3r_live['type_miction']  = '---'
    atoms3r_live['labels']        = []
    return jsonify({'success': True}), 200

# ── Buffer pour logs Serial ───────────────────────────────────
serial_logs = []

@app.route('/api/serial/log', methods=['POST'])
def serial_log():
    data = request.get_json()
    if data and 'msg' in data:
        serial_logs.append({
            'ts':  datetime.now().strftime('%H:%M:%S'),
            'msg': data['msg']
        })
        if len(serial_logs) > 500:
            serial_logs.pop(0)
    return jsonify({'success': True})

@app.route('/api/serial/logs', methods=['GET'])
def get_serial_logs():
    return jsonify({'logs': serial_logs[-100:]})

@app.route('/api/serial/clear', methods=['POST'])
def clear_serial_logs():
    serial_logs.clear()
    return jsonify({'success': True})
    
    # ── Routes Historique ──────────────────────────────────────────
@app.route('/api/historique', methods=['GET'])
def historique():
    try:
        limit = request.args.get('limit', 50, type=int)
        return jsonify({'success': True, 'bilans': get_historique(limit)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historique/stats', methods=['GET'])
def historique_stats():
    try:
        return jsonify({'success': True, **get_stats()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historique/activite', methods=['GET'])
def historique_activite():
    try:
        return jsonify({'success': True, **get_activite()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historique/<int:bilan_id>', methods=['DELETE'])
def historique_supprimer(bilan_id):
    try:
        supprimer_bilan(bilan_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


