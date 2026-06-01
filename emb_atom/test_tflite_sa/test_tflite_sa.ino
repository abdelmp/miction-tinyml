#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <M5Unified.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "arduinoFFT.h"
#include "cnn_miction_model_v3.h"

// ── Configuration WiFi ────────────────────────────────────────
const char* WIFI_SSID = "IoT";     
const char* WIFI_PASS = "";  
const char* SERVER_URL = "http://172.14.3.4:5000/api/atoms3r/bilan"; // ← IP PC

// ── États ─────────────────────────────────────────────────────
enum SystemState { ATTENTE, ENREGISTREMENT, ANALYSE, RESULTATS, SERIAL_TEST };
SystemState currentState = ATTENTE;

// ── TFLite ────────────────────────────────────────────────────
const int kTensorArenaSize = 1300 * 1024;
uint8_t*  tensor_arena     = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// ── Audio ─────────────────────────────────────────────────────
#define SAMPLE_RATE   16000
#define SEG_SAMPLES   8000    // 500ms par segment
#define BYTES_PER_SEC (SAMPLE_RATE * sizeof(int16_t))  // 32KB/s
#define MAX_PSRAM_AUDIO (5 * 1024 * 1024)  // 5MB pour audio
#define MAX_SAMPLES   (MAX_PSRAM_AUDIO / sizeof(int16_t))
#define MAX_SEGMENTS  (MAX_SAMPLES / SEG_SAMPLES)  // ~437 segments = ~218s

#define N_MELS   64
#define N_FFT    512
#define HOP_LEN  160
#define N_FRAMES 51
#define N_FREQ   (N_FFT/2 + 1)

// ── Buffers PSRAM ─────────────────────────────────────────────
int16_t* audio_buffer   = nullptr;  // enregistrement brut
float*   mel_filterbank = nullptr;

// ── Compteurs ─────────────────────────────────────────────────
int   samples_recorded = 0;   // nb samples enregistrés
int   segs_recorded    = 0;   // nb segments complets
unsigned long t_debut  = 0;

// ── Résultats ─────────────────────────────────────────────────
const char* LABELS[] = {"ambiant","chasse","miction"};
uint16_t    COLORS[] = {ORANGE,   CYAN,    GREEN};

float duree_miction = 0.0f;
int   n_episodes    = 0;
bool  chasse        = false;
int   segs_analyses = 0;
bool mode_serial = false;  // ← variable globale


#define MAX_HISTORY 500
int8_t label_history[MAX_HISTORY];
int    history_len = 0;

// ════════════════════════════════════════════════════════════════
// Filtres Mel
// ════════════════════════════════════════════════════════════════
void initMelFilterbank() {
    mel_filterbank = (float*)heap_caps_malloc(
        N_MELS * N_FREQ * sizeof(float), MALLOC_CAP_SPIRAM);
    memset(mel_filterbank, 0, N_MELS * N_FREQ * sizeof(float));

    auto hz2mel = [](float hz){
        return 2595.0f * log10f(1.0f + hz / 700.0f); };
    auto mel2hz = [](float mel){
        return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); };

    float mel_min  = hz2mel(0.0f);
    float mel_max  = hz2mel(6000.0f);
    float mel_step = (mel_max - mel_min) / (N_MELS + 1);

    float mel_pts[N_MELS+2], freq_pts[N_MELS+2];
    for (int i = 0; i < N_MELS+2; i++) {
        mel_pts[i]  = mel_min + i * mel_step;
        freq_pts[i] = mel2hz(mel_pts[i]) * N_FFT / SAMPLE_RATE;
    }

    for (int m = 0; m < N_MELS; m++) {
        float fl = freq_pts[m];
        float fc = freq_pts[m+1];
        float fr = freq_pts[m+2];

        // Construire filtre triangulaire
        float enorm = 0.0f;
        for (int k = 0; k < N_FREQ; k++) {
            float f = (float)k;
            float val = 0.0f;
            if      (f >= fl && f <= fc) val = (f-fl)/(fc-fl+1e-9f);
            else if (f >  fc && f <= fr) val = (fr-f)/(fr-fc+1e-9f);
            mel_filterbank[m*N_FREQ+k] = val;
            enorm += val;
        }

        /*/ ── Normalisation Slaney (identique à librosa) ────────
        // Diviser par la largeur en Hz du filtre
        float hz_fl = mel2hz(mel_pts[m]);
        float hz_fr = mel2hz(mel_pts[m+2]);
        float norm  = 2.0f / (hz_fr - hz_fl + 1e-9f);

        for (int k = 0; k < N_FREQ; k++)
            mel_filterbank[m*N_FREQ+k] *= norm;*/
        

        
    }

}
// ════════════════════════════════════════════════════════════════
// Spectrogramme Mel
// ════════════════════════════════════════════════════════════════
void computeMelSpec(int16_t* segment, float* mel_out) {
    static float vReal[N_FFT];
    static float vImag[N_FFT];
    ArduinoFFT<float> FFT;
    memset(mel_out, 0, N_MELS * N_FRAMES * sizeof(float));

    // Correction glitches
    int64_t sum_sq = 0;
    for (int i = 0; i < SEG_SAMPLES; i++)
        sum_sq += (int64_t)segment[i] * segment[i];
    float rms   = sqrtf((float)sum_sq / SEG_SAMPLES);
    float seuil = rms * 20.0f;
    for (int i = 0; i < SEG_SAMPLES; i++)
        if (abs(segment[i]) > seuil) segment[i] = 0;

    float mel_min_val =  1e10f, mel_max_val = -1e10f;

        for (int t = 0; t < N_FRAMES; t++) {
            int start = t * HOP_LEN;
            if (start + N_FFT > SEG_SAMPLES) {
                // Remplir les frames manquantes avec zéro
                for (int m = 0; m < N_MELS; m++)
                    mel_out[m*N_FRAMES+t] = mel_min_val;  // valeur min
                continue;  // ← continue au lieu de break
            }
        for (int i = 0; i < N_FFT; i++) {
            float hann = 0.5f*(1.0f-cosf(2.0f*M_PI*i/(N_FFT)));
            vReal[i] = ((float)segment[start+i]/32768.0f)*hann;
            vImag[i] = 0.0f;
        }
        FFT.compute(vReal, vImag, N_FFT, FFT_FORWARD);
        FFT.complexToMagnitude(vReal, vImag, N_FFT);

         // ── Correction fréquentielle ──────────────────────────

         // Variable globale
       
        for (int k = 0; k < N_FREQ; k++) {
            float freq = (float)k * SAMPLE_RATE / N_FFT;
            float correction = 1.0f;
        
           /* if      (freq < 500)                  correction = 0.50f;
            else if (freq >= 500  && freq < 1000) correction = 1.40f;
            else if (freq >= 1000 && freq < 2000) correction = 1.15f;
            else if (freq >= 2000 && freq < 4000) correction = 0.25f;
            else if (freq >= 4000)                correction = 0.50f;*/

            vReal[k] *= correction;
        }
        


        for (int m = 0; m < N_MELS; m++) {
            float val = 0.0f;
            for (int k = 0; k < N_FREQ; k++)
            val += mel_filterbank[m*N_FREQ+k] * vReal[k];  // ← sans *vReal[k]
            //val += mel_filterbank[m*N_FREQ+k] * vReal[k] * vReal[k];

            val = 10.0f * log10f(val + 1e-10f);
            mel_out[m*N_FRAMES+t] = val;
            if (val < mel_min_val) mel_min_val = val;
            if (val > mel_max_val) mel_max_val = val;
        }
    }
        float range = mel_max_val - mel_min_val;
        if (range > 0) {
            for (int i = 0; i < N_MELS*N_FRAMES; i++) {
                float val = (mel_out[i] - mel_min_val) / range;
                // Clip entre 0 et 1 — sécurité
                if (val < 0.0f) val = 0.0f;
                if (val > 1.0f) val = 1.0f;
                mel_out[i] = val;
            }
        } else {
            // Signal plat → tout à zéro
            memset(mel_out, 0, N_MELS*N_FRAMES*sizeof(float));
        }
}

// ════════════════════════════════════════════════════════════════
// Inférence CNN
// ════════════════════════════════════════════════════════════════
//int predireSegment(float* mel_spec) {
int predireSegment(float* mel_spec, bool debug=false) {
    TfLiteTensor* input = interpreter->input(0);
    float   scale = input->params.scale;
    int32_t zp    = input->params.zero_point;
    for (int i = 0; i < N_MELS*N_FRAMES; i++)
        input->data.int8[i] = (int8_t)roundf(mel_spec[i]/scale+zp);
    interpreter->Invoke();
    TfLiteTensor* output = interpreter->output(0);

// ── Debug scores ──────────────────────────────────────
    if (debug) {
        float out_scale = output->params.scale;
        int32_t out_zp  = output->params.zero_point;
        Serial.printf("  scores → amb:%.3f cha:%.3f mic:%.3f\n",
            (output->data.int8[0]-out_zp)*out_scale,
            (output->data.int8[1]-out_zp)*out_scale,
            (output->data.int8[2]-out_zp)*out_scale);
    }

    int8_t max_val = -128; int max_idx = 0;
    for (int i = 0; i <3; i++)
        if (output->data.int8[i] > max_val) {
            max_val = output->data.int8[i]; max_idx = i; }
    return max_idx;
}

// ════════════════════════════════════════════════════════════════
// Corection sur les segments
// ════════════════════════════════════════════════════════════
void corrigerLabels() {
    if (history_len < 3) return;

    // ── Passe 1 : corriger_contexte (fenetre=1) ───────────────
    // Si labels[i-1] == labels[i+1] != labels[i] → labels[i] = labels[i-1]
    for (int i = 1; i < history_len - 1; i++) {
        if (label_history[i-1] == label_history[i+1] &&
            label_history[i]   != label_history[i-1]) {
            label_history[i] = label_history[i-1];
        }
    }

    // ── Passe 2 : fusion trous miction 1-3 segments ───────────
    // Identique à la boucle gap*3 du pipeline Python
    int gap = 4;
    for (int iter = 0; iter < gap * 3; iter++) {
        bool changed = false;
        for (int trou = 1; trou <= 3; trou++) {
            for (int i = 0; i < history_len - trou - 1; i++) {
                if (label_history[i] == 2 &&
                    label_history[i + trou + 1] == 2) {
                    // Vérifier que tous les segments du trou != miction
                    bool all_non_mic = true;
                    for (int k = 1; k <= trou; k++) {
                        if (label_history[i+k] == 2) {
                            all_non_mic = false;
                            break;
                        }
                    }
                    if (all_non_mic) {
                        for (int k = 1; k <= trou; k++)
                            label_history[i+k] = 2;
                        changed = true;
                    }
                }
            }
        }
        if (!changed) break;
    }

    // ── Passe 3 : supprimer segments isolés (vote majoritaire) ─
    // Identique à la passe 3 Python — fenêtre ±2
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 1; i < history_len - 1; i++) {

            // Construire contexte fenêtre ±2 (sans i)
            int contexte[4];
            int ctx_len = 0;
            for (int k = max(0, i-2); k < min(history_len, i+3); k++) {
                if (k != i) contexte[ctx_len++] = label_history[k];
            }

            // Compter votes pour chaque classe
            int count[3] = {0, 0, 0};
            for (int c = 0; c < ctx_len; c++) {
                if (contexte[c] >= 0 && contexte[c] < 3)
                    count[contexte[c]]++;
            }

            // Trouver le vote majoritaire
            int vote = 0;
            for (int c = 1; c < 3; c++)
                if (count[c] > count[vote]) vote = c;

            // Compter combien de fois label_history[i] apparaît dans contexte
            int count_i = count[label_history[i]];

            // Si isolé (≤1 voisin identique) ET vote différent → remplacer
            if (count_i <= 1 && vote != label_history[i]) {
                label_history[i] = vote;
                changed = true;
            }
        }
    }
}
// ════════════════════════════════════════════════════════════════
// Calcul indicateurs finaux
// ════════════════════════════════════════════════════════════════
void calculerIndicateurs() {
    corrigerLabels();  
    int n_mic = 0;
    chasse = false;
    for (int i = 0; i < history_len; i++) {
        if (label_history[i] == 2) n_mic++;
        if (label_history[i] == 1) chasse = true;
    }
    duree_miction = n_mic * 0.5f;

    n_episodes = 0;
    bool en_miction = false;
    for (int i = 0; i < history_len; i++) {
        if (label_history[i] == 2 && !en_miction) {
            en_miction = true; n_episodes++;
        } else if (label_history[i] != 2) {
            en_miction = false;
        }
    }
}

String getType() {
    if      (duree_miction < 8.0f) return "jet_faible";
    else if (n_episodes >= 2)       return "jet_hache";
    else                            return "jet_continu";
}

// ── Connexion WiFi ────────────────────────────────────────────
void connectWiFi() {
    M5.Display.setTextColor(WHITE);
    M5.Display.println("WiFi...");
    M5.Display.printf("SSID: %s\n", WIFI_SSID);
    
    WiFi.mode(WIFI_STA);  // ← ajouter mode station
    WiFi.disconnect();     // ← reset WiFi avant
    delay(100);
    
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    int tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries < 30) {
        delay(500);
        tries++;
        M5.Display.print(".");
        Serial.printf("Tentative %d — status: %d\n", tries, WiFi.status());
    }
    if (WiFi.status() == WL_CONNECTED) {
        M5.Display.setTextColor(GREEN);
        M5.Display.printf("\nWiFi OK\n%s\n",
            WiFi.localIP().toString().c_str());
        Serial.printf("WiFi OK — IP: %s\n",
            WiFi.localIP().toString().c_str());
    } else {
        M5.Display.setTextColor(RED);
        M5.Display.printf("\nWiFi FAIL\nStatus: %d\n", WiFi.status());
        Serial.printf("WiFi FAIL — status final: %d\n", WiFi.status());
    }
    delay(1000);
}
// ════════════════════════════════════════════════════════════════
// Écrans
// ════════════════════════════════════════════════════════════════
void afficherAttente() {
    M5.Display.fillScreen(BLUE);
    M5.Display.setTextColor(WHITE);
    M5.Display.setTextSize(1);
    M5.Display.setCursor(5, 10);
    M5.Display.println(" PROTO CHU ANGERS");
    M5.Display.drawFastHLine(0, 22, 128, WHITE);
    M5.Display.setCursor(5, 30);
    M5.Display.setTextColor(0xAEFF);  // vert clair
    M5.Display.printf("PSRAM: %dKB libres\n",
        heap_caps_get_free_size(MALLOC_CAP_SPIRAM)/1024);
    M5.Display.printf("Capacite: ~%ds\n",
        (int)(MAX_PSRAM_AUDIO / BYTES_PER_SEC));
    M5.Display.drawFastHLine(0, 55, 128, WHITE);
    M5.Display.setTextColor(WHITE);
    M5.Display.setTextSize(2);
    M5.Display.setCursor(10, 65);
    M5.Display.println("APPUYER");
    M5.Display.setCursor(15, 88);
    M5.Display.println("POUR");
    M5.Display.setCursor(10, 110);
    M5.Display.println("ENREG.");
}

void afficherEnregistrement() {
    unsigned long t_s = (millis() - t_debut) / 1000;
    int pct = (samples_recorded * 100) / MAX_SAMPLES;

    M5.Display.fillRect(0, 15, 128, 95, BLACK);
    M5.Display.setCursor(0, 15);
    M5.Display.setTextColor(RED);
    M5.Display.printf("REC %ds\n", (int)t_s);
    M5.Display.setTextColor(WHITE);
    M5.Display.printf("Segs: %d\n", segs_recorded);
    M5.Display.printf("PSRAM: %d%%\n", pct);

    // Barre PSRAM
    M5.Display.fillRect(0, 65, 128, 10, TFT_DARKGREY);
    M5.Display.fillRect(0, 65, pct*128/100, 10,
        pct > 80 ? RED : pct > 50 ? ORANGE : GREEN);

    M5.Display.setCursor(0, 80);
    M5.Display.setTextColor(TFT_DARKGREY);
    M5.Display.println("Appuyer pour arreter");
}

void afficherAnalyse(int seg_en_cours, int seg_total) {
    int pct = (seg_en_cours * 100) / max(seg_total, 1);
    M5.Display.fillRect(0, 15, 128, 60, BLACK);
    M5.Display.setCursor(0, 15);
    M5.Display.setTextColor(CYAN);
    M5.Display.printf("ANALYSE %d%%\n", pct);
    M5.Display.setTextColor(WHITE);
    M5.Display.printf("Seg %d/%d\n", seg_en_cours, seg_total);
    M5.Display.printf("Miction: %.1fs\n", duree_miction);

    // Barre progression
    M5.Display.fillRect(0, 75, 128, 10, TFT_DARKGREY);
    M5.Display.fillRect(0, 75, pct*128/100, 10, CYAN);
}

void afficherResultats() {
    M5.Display.fillScreen(BLACK);
    M5.Display.drawRect(0, 0, 128, 128, WHITE);
    M5.Display.setTextSize(1);

    M5.Display.setTextColor(YELLOW);
    M5.Display.setCursor(20, 4);
    M5.Display.println("BILAN FINAL");
    M5.Display.drawFastHLine(5, 15, 118, WHITE);

    M5.Display.setTextColor(WHITE);
    M5.Display.setCursor(5, 19);
    M5.Display.printf("Enreg : %ds (%d segs)\n",
        (int)(segs_recorded * 0.5f), segs_recorded);
    M5.Display.drawFastHLine(5, 32, 118, TFT_DARKGREY);

    M5.Display.setCursor(5, 36);
    M5.Display.setTextColor(GREEN);
    M5.Display.printf("Miction : %.1fs\n", duree_miction);
    M5.Display.setTextColor(WHITE);
    M5.Display.setCursor(5, 50);
    M5.Display.printf("Type    : %s\n", getType().c_str());
    M5.Display.setCursor(5, 64);
    M5.Display.printf("Episodes: %d\n", n_episodes);
    M5.Display.setCursor(5, 78);
    M5.Display.printf("Chasse  : %s\n", chasse ? "OUI" : "NON");
    M5.Display.drawFastHLine(5, 94, 118, WHITE);

    M5.Display.setTextColor(TFT_DARKGREY);
    M5.Display.setCursor(5, 98);
    M5.Display.printf("Segs analyses: %d\n", segs_analyses);

    M5.Display.setTextColor(GREEN);
    M5.Display.setCursor(5, 114);
    M5.Display.println("> Clic pour RAZ");
}
// ── Envoi update temps réel (1 segment) ──────────────────────
void envoyerUpdate(int seg_idx, int label_idx, float duree, String type) {
    if (WiFi.status() != WL_CONNECTED) return;

    String json = "{";
    json += "\"seg\":" + String(seg_idx) + ",";
    json += "\"label\":" + String(label_idx) + ",";
    json += "\"duree_miction\":" + String(duree, 1) + ",";
    json += "\"type_miction\":\"" + type + "\",";
    json += "\"total_segs\":" + String(segs_recorded);
    json += "}";

    HTTPClient http;
    http.begin("http://172.14.3.4:5000/api/atoms3r/update");
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(300);  // timeout court — 300ms max
    http.POST(json);
    http.end();
}


// ── Nouvelle fonction réception Serial ────────────────────
void recevoirEtAnalyserSerial() {

    samples_recorded = 0;
    segs_recorded    = 0;
    history_len      = 0;

    logMsg("MODE TEST SERIAL — en attente AUDIO:N\n");

    while (!Serial.available()) delay(10);
    String header = Serial.readStringUntil('\n');
    header.trim();

    if (!header.startsWith("AUDIO:")) {
        logMsg("ERR: header invalide\n");
        return;
    }

    int n_segs = header.substring(6).toInt();
    logMsg("Reception %d segments...\n", n_segs);

    for (int s = 0; s < n_segs && s < MAX_SEGMENTS; s++) {
        int16_t* dest    = audio_buffer + s * SEG_SAMPLES;
        int bytes_needed = SEG_SAMPLES * sizeof(int16_t);
        int bytes_read   = 0;
        while (bytes_read < bytes_needed) {
            int avail = Serial.available();
            if (avail > 0) {
                int to_read = min(avail, bytes_needed - bytes_read);
                Serial.readBytes((char*)dest + bytes_read, to_read);
                bytes_read += to_read;
            }
            
        }
        samples_recorded += SEG_SAMPLES;
        segs_recorded     = s + 1;
        if (s % 10 == 0)
            Serial.printf("Recu seg %d/%d\n", s+1, n_segs);
    }

    logMsg("Reception OK — %d segments\n", segs_recorded);
    logMsg("=== DEBUT ANALYSE ===\n");

    // Normalisation globale
    int16_t global_max = 0;
    for (int i = 0; i < samples_recorded; i++) {
        int16_t v = abs(audio_buffer[i]);
        if (v > global_max) global_max = v;
    }
    logMsg("Max global: %d\n", global_max);
    if (global_max > 0) {
        for (int i = 0; i < samples_recorded; i++)
            audio_buffer[i] = (int16_t)(
                (float)audio_buffer[i] / global_max * 32767.0f);
    }

    duree_miction = 0; n_episodes = 0;
    chasse = false; segs_analyses = 0; history_len = 0;

    static float mel_spec[N_MELS * N_FRAMES];

    for (int s = 0; s < segs_recorded; s++) {
        int16_t* src = audio_buffer + s * SEG_SAMPLES;
        computeMelSpec(src, mel_spec);

        bool debug_seg = (s % 5 == 0);
        int label_idx  = predireSegment(mel_spec, debug_seg);

        if (s % 5 == 0) {
            float sum=0, mx=-1, mn=2;
            for (int i = 0; i < N_MELS*N_FRAMES; i++) {
                sum += mel_spec[i];
                if (mel_spec[i] > mx) mx = mel_spec[i];
                if (mel_spec[i] < mn) mn = mel_spec[i];
            }
            logMsg("Seg %3d | min=%.3f mean=%.3f max=%.3f | %s\n",
                s, mn, sum/(N_MELS*N_FRAMES), mx, LABELS[label_idx]);
        } else {
            logMsg("Seg %3d → %s\n", s, LABELS[label_idx]);
        }

        if (history_len < MAX_HISTORY)
            label_history[history_len++] = label_idx;
        segs_analyses++;
    }

    calculerIndicateurs();

    logMsg("=== RESULTATS ===\n");
    logMsg("Duree miction : %.1f s\n", duree_miction);
    logMsg("Type          : %s\n",     getType().c_str());
    logMsg("Episodes      : %d\n",     n_episodes);
    logMsg("Chasse        : %s\n",     chasse ? "OUI" : "NON");
    logMsg("Segs analyses : %d\n",     segs_analyses);

    currentState = RESULTATS;
    afficherResultats();


}


// ════════════════════════════════════════════════════════════════
// Envoi bilan vers Flask
// ════════════════════════════════════════════════════════════════
void envoyerBilan() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi non connecte");
        M5.Display.setTextColor(ORANGE);
        M5.Display.println("WiFi OFF");
        return;
    }

    M5.Display.setTextColor(CYAN);
    M5.Display.println("Envoi WiFi...");

    // Construire JSON avec labels
    String labels_str = "[";
    for (int i = 0; i < history_len; i++) {
        labels_str += String(label_history[i]);
        if (i < history_len - 1) labels_str += ",";
    }
    labels_str += "]";

    String json = "{";
    json += "\"duree_miction\":" + String(duree_miction, 1) + ",";
    json += "\"type_miction\":\"" + getType() + "\",";
    json += "\"n_episodes\":" + String(n_episodes) + ",";
    json += "\"chasse\":" + String(chasse ? "true" : "false") + ",";
    json += "\"segs_analyses\":" + String(segs_analyses) + ",";
    json += "\"labels\":" + labels_str;
    json += "}";

    Serial.printf("Envoi JSON %d bytes\n", json.length());

    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(10000);

    int code = http.POST(json);

    if (code == 200) {
        M5.Display.setTextColor(GREEN);
        M5.Display.println("Envoye OK!");
        Serial.println("Bilan envoye OK");
    } else {
        M5.Display.setTextColor(RED);
        M5.Display.printf("HTTP ERR: %d\n", code);
        Serial.printf("HTTP ERR: %d\n", code);
    }
    http.end();
    delay(1000);
}

// ── Log vers Serial ET Flask ──────────────────────────────────
void logMsg(const char* format, ...) {
    char buf[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);

    // Toujours afficher en Serial
    Serial.print(buf);

    // Envoyer vers Flask si WiFi connecté
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin("http://172.14.3.4:5000/api/serial/log");
        http.addHeader("Content-Type", "application/json");
        http.setTimeout(200);
        String json = "{\"msg\":\"";
        // Échapper les guillemets
        String msg = String(buf);
        msg.replace("\"", "'");
        msg.replace("\n", " ");
        json += msg + "\"}";
        http.POST(json);
        http.end();
    }
}
// ════════════════════════════════════════════════════════════════
// SETUP
// ════════════════════════════════════════════════════════════════
void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    Serial.setRxBufferSize(65536);
    Serial.begin(921600);

    M5.Display.setTextSize(1);
    M5.Display.fillScreen(BLACK);
    M5.Display.setTextColor(WHITE);
    M5.Display.println("IA Miction CHU");
    M5.Display.println("Initialisation...");

    // Buffer audio en PSRAM
    audio_buffer = (int16_t*)heap_caps_malloc(
        MAX_PSRAM_AUDIO, MALLOC_CAP_SPIRAM);
    if (!audio_buffer) {
        M5.Display.setTextColor(RED);
        M5.Display.println("Audio buffer ERR!");
        while(1) delay(100);
    }
    M5.Display.printf("Audio OK (%ds max)\n",
        (int)(MAX_PSRAM_AUDIO / BYTES_PER_SEC));

    // TFLite
    tensor_arena = (uint8_t*)heap_caps_malloc(
        kTensorArenaSize, MALLOC_CAP_SPIRAM);
    if (!tensor_arena) {
        M5.Display.setTextColor(RED);
        M5.Display.println("TFLite PSRAM ERR!");
        while(1) delay(100);
    }

    const tflite::Model* model_tfl = tflite::GetModel(cnn_miction_model_v3);
    static tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMean();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    static tflite::MicroInterpreter static_interpreter(
        model_tfl, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        M5.Display.setTextColor(RED);
        M5.Display.println("AllocTensors ERR!");
        while(1) delay(100);
    }
    M5.Display.println("TFLite OK");

    // Filtres Mel
    initMelFilterbank();
    M5.Display.println("Mel OK");

    // Microphone
    auto mic_cfg = M5.Mic.config();
    mic_cfg.sample_rate   = SAMPLE_RATE;
    mic_cfg.pin_bck       = 7;
    mic_cfg.pin_ws        = 6;
    mic_cfg.pin_data_in   = 5;
    mic_cfg.i2s_port      = I2S_NUM_0;
    mic_cfg.magnification = 1;
    mic_cfg.stereo        = false;
    mic_cfg.left_channel  = true;
    M5.Mic.config(mic_cfg);
    M5.Mic.begin();
    // Test microphone au démarrage
    M5.Display.println("Test micro...");
    int16_t test_buf[1600];  // 100ms
    bool ok = M5.Mic.record(test_buf, 1600, SAMPLE_RATE);

    int16_t test_max = 0;
    for (int i = 0; i < 1600; i++)
        if (abs(test_buf[i]) > test_max)
            test_max = abs(test_buf[i]);

    Serial.printf("Test micro: ok=%d max=%d\n", ok, test_max);
    logMsg("Test micro: ok=%d max=%d\n", ok, test_max);

    if (test_max < 100) {
        M5.Display.setTextColor(RED);
        M5.Display.println("MICRO FAIBLE!");
    } else {
        M5.Display.setTextColor(GREEN);
        M5.Display.printf("MICRO OK max=%d\n", test_max);
    }
    delay(1000);
   // M5.Display.println("Micro OK");

    // ── Connexion WiFi ────────────────────────────────────────────
    M5.Display.setTextColor(WHITE);
    M5.Display.println("WiFi...");
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    delay(100);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    int tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries < 30) {
        delay(500);
        tries++;
        M5.Display.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) {
        M5.Display.setTextColor(GREEN);
        M5.Display.printf("\nWiFi OK\n%s\n",
            WiFi.localIP().toString().c_str());
        Serial.printf("WiFi OK — IP: %s\n",
            WiFi.localIP().toString().c_str());
    } else {
        M5.Display.setTextColor(RED);
        M5.Display.println("\nWiFi FAIL");
    }
    delay(1000);

    // Test ping Flask
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin("http://172.14.3.4:5000/api/serial/log");
        http.addHeader("Content-Type", "application/json");
        int code = http.POST("{\"msg\":\"ATOMS3R connecte et pret\"}");
        Serial.printf("Test Flask : HTTP %d\n", code);
        http.end();
    }
}

// ════════════════════════════════════════════════════════════════
// LOOP
// ════════════════════════════════════════════════════════════════
void loop() {
    M5.update();

    switch (currentState) {

        // ── ATTENTE ───────────────────────────────────────────
        case ATTENTE:
            if (Serial.available()) {
                String cmd = Serial.readStringUntil('\n');
                cmd.trim();
                if (cmd == "TEST") {
                    recevoirEtAnalyserSerial();
                }
            }
            if (M5.BtnA.wasPressed()) {
                samples_recorded = 0;
                segs_recorded    = 0;
                t_debut          = millis();
                currentState     = ENREGISTREMENT;

                M5.Display.fillScreen(BLACK);
                M5.Display.setTextSize(1);
                M5.Display.setTextColor(RED);
                M5.Display.setCursor(0, 0);
                M5.Display.println("● ENREGISTREMENT");
                M5.Display.drawFastHLine(0, 12, 128, WHITE);
            }
            break;

        // ── ENREGISTREMENT ────────────────────────────────────
        case ENREGISTREMENT: {
            // Arrêt manuel ou PSRAM pleine
            if (M5.BtnA.wasPressed() ||
                samples_recorded + SEG_SAMPLES > MAX_SAMPLES) {
                currentState = ANALYSE;

                M5.Display.fillScreen(BLACK);
                M5.Display.setTextSize(1);
                M5.Display.setTextColor(CYAN);
                M5.Display.setCursor(0, 0);
                M5.Display.println("ANALYSE EN COURS...");
                M5.Display.drawFastHLine(0, 12, 128, WHITE);
                break;
            }

            // Enregistrer directement dans PSRAM
            int16_t* dest = audio_buffer + samples_recorded;
            if (M5.Mic.record(dest, SEG_SAMPLES, SAMPLE_RATE)) {
                samples_recorded += SEG_SAMPLES;
                segs_recorded++;
                afficherEnregistrement();
            }
            break;
        }

        // ── ANALYSE ───────────────────────────────────────────
        case ANALYSE: {
            // Normalisation globale
            int16_t global_max = 0;
            for (int i = 0; i < samples_recorded; i++) {
                int16_t v = abs(audio_buffer[i]);
                if (v > global_max) global_max = v;
            }
            logMsg("Max global signal: %d\n", global_max);
            if (global_max > 0) {
                for (int i = 0; i < samples_recorded; i++)
                    audio_buffer[i] = (int16_t)(
                        (float)audio_buffer[i] / global_max * 32767.0f);
            }

             // Après normalisation globale
            logMsg("Signal brut: max=%d segs=%d\n", global_max, segs_recorded);

            // Afficher RMS des 5 premiers segments
            for (int s = 0; s < min(5, segs_recorded); s++) {
                int16_t* src = audio_buffer + s * SEG_SAMPLES;
                int64_t sum_sq = 0;
                for (int i = 0; i < SEG_SAMPLES; i++)
                    sum_sq += (int64_t)src[i] * src[i];
                float rms = sqrtf((float)sum_sq / SEG_SAMPLES);
                logMsg("Seg %d RMS = %.1f\n", s, rms);
            }

            duree_miction = 0; n_episodes = 0;
            chasse = false; segs_analyses = 0; history_len = 0;

            static float mel_spec[N_MELS * N_FRAMES];

          /* for (int s = 0; s < segs_recorded; s++) {
                int16_t* src = audio_buffer + s * SEG_SAMPLES;
                computeMelSpec(src, mel_spec);

                bool debug_seg = (s % 5 == 0);
                int label_idx  = predireSegment(mel_spec, debug_seg);

                // ── DEBUG — 1 ligne propre par segment ───────────────
                if (s % 5 == 0) {
                    float sum=0, mx=-1, mn=2;
                    for (int i = 0; i < N_MELS*N_FRAMES; i++) {
                        sum += mel_spec[i];
                        if (mel_spec[i] > mx) mx = mel_spec[i];
                        if (mel_spec[i] < mn) mn = mel_spec[i];
                    }
                    logMsg(
                        "Seg %3d | min=%.3f mean=%.3f max=%.3f | %s\n",
                        s, mn, sum/(N_MELS*N_FRAMES), mx, LABELS[label_idx]);
                } else {
                    logMsg("Seg %3d → %s\n", s, LABELS[label_idx]);
                }

                if (history_len < MAX_HISTORY)
                    label_history[history_len++] = label_idx;
                segs_analyses++;

                int n_mic_tmp = 0;
                for (int i = 0; i < history_len; i++)
                    if (label_history[i] == 2) n_mic_tmp++;
                float duree_tmp = n_mic_tmp * 0.5f;

                if (s % 3 == 0)
                    envoyerUpdate(s, label_idx, duree_tmp, getType());

                if (s % 5 == 0) {
                    duree_miction = duree_tmp;
                    afficherAnalyse(s+1, segs_recorded);
                    M5.update();
                }
            }*/
            // Debug complet — afficher TOUS les segments
            for (int s = 0; s < segs_recorded; s++) {
                int16_t* src = audio_buffer + s * SEG_SAMPLES;
                
                // RMS du segment brut
                int64_t sum_sq = 0;
                for (int i = 0; i < SEG_SAMPLES; i++)
                    sum_sq += (int64_t)src[i] * src[i];
                float rms = sqrtf((float)sum_sq / SEG_SAMPLES);
                
                computeMelSpec(src, mel_spec);
                bool debug_seg = true;  // ← debug sur TOUS les segments
                int label_idx = predireSegment(mel_spec, debug_seg);
                
                logMsg("Seg %2d RMS=%.0f → %s\n", s, rms, LABELS[label_idx]);
                
                if (history_len < MAX_HISTORY)
                    label_history[history_len++] = label_idx;
                segs_analyses++;
            }
            calculerIndicateurs();
            envoyerBilan();
            currentState = RESULTATS;
            break;
        }
        // ── RESULTATS ─────────────────────────────────────────
        case RESULTATS:
            afficherResultats();
            while (currentState == RESULTATS) {
                M5.update();
                if (M5.BtnA.wasPressed()) {
                    currentState = ATTENTE;
                    afficherAttente();
                }
                delay(50);
            }
            break;
    }
}