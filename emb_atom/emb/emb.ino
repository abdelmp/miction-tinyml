#include <M5Unified.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "arduinoFFT.h"
#include "cnn_miction_model.h"

// ── TFLite ────────────────────────────────────────────────────
const int kTensorArenaSize = 1300 * 1024;
uint8_t* tensor_arena      = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// ── Audio ─────────────────────────────────────────────────────
#define SAMPLE_RATE  16000
#define SEG_SAMPLES  8000    // 500ms
#define N_MELS       64
#define N_FFT        512
#define HOP_LEN      160
#define N_FRAMES     51      // floor((8000 - 512) / 160) + 1

// Labels
const char* LABELS[] = {
    "bruit_ambiant", "bruit_chasse", "miction_active", "silence"
};

// Indicateurs médicaux
float  duree_miction = 0.0f;
int    n_episodes    = 0;
String type_miction  = "---";
bool   chasse        = false;
int    n_segs_total  = 0;
int    n_segs_miction= 0;

// Historique labels pour lissage
#define MAX_HISTORY 200
int8_t label_history[MAX_HISTORY];
int    history_len = 0;

// ── Filtres Mel (pré-calculés) ────────────────────────────────
// Calculés une fois au démarrage depuis PSRAM
float* mel_filterbank = nullptr;  // (N_MELS × (N_FFT/2+1))
#define N_FREQ  (N_FFT/2 + 1)    // 257

void computeMelFilterbank() {
    // Allouer en PSRAM
    mel_filterbank = (float*)heap_caps_malloc(
        N_MELS * N_FREQ * sizeof(float), MALLOC_CAP_SPIRAM);
    memset(mel_filterbank, 0, N_MELS * N_FREQ * sizeof(float));

    // Conversion Hz → Mel
    auto hz2mel = [](float hz) {
        return 2595.0f * log10f(1.0f + hz / 700.0f);
    };
    auto mel2hz = [](float mel) {
        return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    };

    float fmax     = 6000.0f;
    float mel_min  = hz2mel(0.0f);
    float mel_max  = hz2mel(fmax);
    float mel_step = (mel_max - mel_min) / (N_MELS + 1);

    // Points centraux des filtres Mel
    float mel_points[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++) {
        mel_points[i] = mel_min + i * mel_step;
    }

    // Fréquences en bins FFT
    float freq_points[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++) {
        freq_points[i] = mel2hz(mel_points[i]) *
                         N_FFT / SAMPLE_RATE;
    }

    // Construire les filtres triangulaires
    for (int m = 0; m < N_MELS; m++) {
        float f_left   = freq_points[m];
        float f_center = freq_points[m + 1];
        float f_right  = freq_points[m + 2];
        for (int k = 0; k < N_FREQ; k++) {
            float f = (float)k;
            if (f >= f_left && f <= f_center) {
                mel_filterbank[m * N_FREQ + k] =
                    (f - f_left) / (f_center - f_left + 1e-9f);
            } else if (f > f_center && f <= f_right) {
                mel_filterbank[m * N_FREQ + k] =
                    (f_right - f) / (f_right - f_center + 1e-9f);
            }
        }
    }
}

// ── Calcul spectrogramme Mel ──────────────────────────────────
// segment (8000 samples) → mel_spec (64 × 51) normalisé [0,1]
// mel_out doit être alloué par l'appelant : float[N_MELS * N_FRAMES]
void computeMelSpec(int16_t* segment, float* mel_out) {
    static float vReal[N_FFT];
    static float vImag[N_FFT];

    ArduinoFFT<float> FFT;

    // Initialiser sortie
    memset(mel_out, 0, N_MELS * N_FRAMES * sizeof(float));

    float mel_min_val =  1e10f;
    float mel_max_val = -1e10f;

    // Pour chaque frame temporelle
    for (int t = 0; t < N_FRAMES; t++) {
        int start = t * HOP_LEN;
        if (start + N_FFT > SEG_SAMPLES) break;

        // Remplir vReal avec fenêtre de Hann
        for (int i = 0; i < N_FFT; i++) {
            float hann = 0.5f * (1.0f - cosf(
                2.0f * M_PI * i / (N_FFT - 1)));
            float s    = (float)segment[start + i] / 32768.0f;
            vReal[i]   = s * hann;
            vImag[i]   = 0.0f;
        }

        // FFT
        FFT.compute(vReal, vImag, N_FFT, FFT_FORWARD);
        FFT.complexToMagnitude(vReal, vImag, N_FFT);

        // Puissance spectrale
        float power[N_FREQ];
        for (int k = 0; k < N_FREQ; k++) {
            power[k] = vReal[k] * vReal[k];
        }

        // Appliquer filtres Mel + log
        for (int m = 0; m < N_MELS; m++) {
            float mel_val = 0.0f;
            for (int k = 0; k < N_FREQ; k++) {
                mel_val += mel_filterbank[m * N_FREQ + k] * power[k];
            }
            // Conversion en dB
            mel_val = 10.0f * log10f(mel_val + 1e-10f);
            mel_out[m * N_FRAMES + t] = mel_val;

            if (mel_val < mel_min_val) mel_min_val = mel_val;
            if (mel_val > mel_max_val) mel_max_val = mel_val;
        }
    }

    // Normalisation [0, 1]
    float range = mel_max_val - mel_min_val;
    if (range > 0) {
        for (int i = 0; i < N_MELS * N_FRAMES; i++) {
            mel_out[i] = (mel_out[i] - mel_min_val) / range;
        }
    }
}

// ── Inférence CNN ─────────────────────────────────────────────
int predireSegment(float* mel_spec) {
    TfLiteTensor* input = interpreter->input(0);

    // Quantifier float [0,1] → int8 [-128, 127]
    float scale     = input->params.scale;
    int32_t zp      = input->params.zero_point;

    for (int m = 0; m < N_MELS; m++) {
        for (int t = 0; t < N_FRAMES; t++) {
            float val = mel_spec[m * N_FRAMES + t];
            int8_t q  = (int8_t)roundf(val / scale + zp);
            // Index : (m, t, 0) dans layout (H, W, C)
            input->data.int8[m * N_FRAMES + t] = q;
        }
    }

    interpreter->Invoke();

    TfLiteTensor* output = interpreter->output(0);
    int8_t max_val = -128;
    int    max_idx = 0;
    for (int i = 0; i < 4; i++) {
        if (output->data.int8[i] > max_val) {
            max_val = output->data.int8[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ── Lissage et indicateurs ────────────────────────────────────
void mettreAJourIndicateurs(int label_idx) {
    // Ajouter à l'historique
    if (history_len < MAX_HISTORY) {
        label_history[history_len++] = label_idx;
    } else {
        memmove(label_history, label_history + 1,
                (MAX_HISTORY - 1) * sizeof(int8_t));
        label_history[MAX_HISTORY - 1] = label_idx;
    }

    // Compter miction_active (index 2)
    n_segs_total  = history_len;
    n_segs_miction = 0;
    chasse         = false;
    for (int i = 0; i < history_len; i++) {
        if (label_history[i] == 2) n_segs_miction++;
        if (label_history[i] == 1) chasse = true; // bruit_chasse
    }

    // Durée miction (hop=250ms)
    duree_miction = n_segs_miction * 0.25f;

    // Compter épisodes
    n_episodes = 0;
    bool en_miction = false;
    for (int i = 0; i < history_len; i++) {
        if (label_history[i] == 2 && !en_miction) {
            en_miction = true;
            n_episodes++;
        } else if (label_history[i] != 2) {
            en_miction = false;
        }
    }

    // Type miction
    if (duree_miction < 8.0f)       type_miction = "jet_faible";
    else if (n_episodes >= 2)        type_miction = "jet_hache";
    else                             type_miction = "jet_continu";
}

// ── Affichage ─────────────────────────────────────────────────
void afficher(int label_idx) {
    M5.Display.fillRect(0, 40, 128, 88, BLACK);
    M5.Display.setCursor(0, 40);

    // Label courant
    uint16_t col = WHITE;
    if      (label_idx == 2) col = GREEN;
    else if (label_idx == 1) col = CYAN;
    else if (label_idx == 3) col = DARKGREY;
    else                     col = ORANGE;

    M5.Display.setTextColor(col);
    M5.Display.println(LABELS[label_idx]);
    M5.Display.setTextColor(WHITE);

    M5.Display.printf("%.1fs %s\n",
        duree_miction, type_miction.c_str());
    M5.Display.printf("Ep:%d Ch:%s\n",
        n_episodes, chasse ? "OUI" : "non");
    M5.Display.printf("Segs:%d\n", n_segs_total);
}

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    M5.Display.setTextSize(1);
    M5.Display.fillScreen(BLACK);
    M5.Display.setTextColor(WHITE);
    M5.Display.println("IA Miction CHU");
    M5.Display.println("Init...");

    // ── PSRAM + TFLite ────────────────────────────────────────
    tensor_arena = (uint8_t*)heap_caps_malloc(
        kTensorArenaSize, MALLOC_CAP_SPIRAM);
    if (!tensor_arena) {
        M5.Display.setTextColor(RED);
        M5.Display.println("PSRAM ERR!");
        while(1) delay(100);
    }

    const tflite::Model* model_tfl = tflite::GetModel(cnn_miction_model);
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
        M5.Display.println("TFLite ERR!");
        while(1) delay(100);
    }
    M5.Display.println("TFLite OK");

    // ── Filtres Mel ───────────────────────────────────────────
    computeMelFilterbank();
    M5.Display.println("Mel OK");

    // ── Microphone ────────────────────────────────────────────
    auto mic_cfg = M5.Mic.config();
    mic_cfg.sample_rate = SAMPLE_RATE;
    mic_cfg.pin_bck     = 1;
    mic_cfg.pin_ws      = 2;
    mic_cfg.pin_data_in = 38;
    M5.Mic.config(mic_cfg);

    if (!M5.Mic.begin()) {
        M5.Display.setTextColor(RED);
        M5.Display.println("Micro ERR!");
        while(1) delay(100);
    }
    M5.Display.println("Micro OK");

    delay(500);
    M5.Display.fillScreen(BLACK);
    M5.Display.setCursor(0, 0);
    M5.Display.println("IA Miction CHU");
    M5.Display.println("En ecoute...");
}

void loop() {
    // Buffer pour 500ms audio (chevauchement 50% → 250ms)
    static int16_t segment[SEG_SAMPLES];
    static float   mel_spec[N_MELS * N_FRAMES];

    // Capturer 500ms
    if (M5.Mic.record(segment, SEG_SAMPLES, SAMPLE_RATE)) {
        // Normaliser
        float mx = 0;
        for (int i = 0; i < SEG_SAMPLES; i++) {
            float v = fabsf((float)segment[i]);
            if (v > mx) mx = v;
        }
        if (mx > 0) {
            for (int i = 0; i < SEG_SAMPLES; i++) {
                segment[i] = (int16_t)((float)segment[i] / mx * 32767.0f);
            }
        }

        // Spectrogramme Mel
        computeMelSpec(segment, mel_spec);

        // Inférence CNN
        int label_idx = predireSegment(mel_spec);

        // Indicateurs
        mettreAJourIndicateurs(label_idx);

        // Affichage
        afficher(label_idx);
    }

    M5.update();
}