#include <M5Unified.h>

#define SAMPLE_RATE   16000
#define MAX_PSRAM     (3 * 1024 * 1024)  // 
#define MAX_SAMPLES   (MAX_PSRAM / sizeof(int16_t))

int16_t* audio_buffer    = nullptr;
int16_t* audio_normalized = nullptr;
int      samples_recorded = 0;
bool     recording        = false;
bool     recorded         = false;

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    Serial.begin(921600);
    delay(2000);

    Serial.println("=== CAPTURE + NORMALISATION ===");

    M5.Display.setTextSize(1);
    M5.Display.fillScreen(BLACK);
    M5.Display.setTextColor(WHITE);
    M5.Display.println("Init buffers PSRAM...");

    // Buffer brut
    audio_buffer = (int16_t*)heap_caps_malloc(
        MAX_PSRAM, MALLOC_CAP_SPIRAM);
    // Buffer normalisé
    audio_normalized = (int16_t*)heap_caps_malloc(
        MAX_PSRAM, MALLOC_CAP_SPIRAM);

    if (!audio_buffer || !audio_normalized) {
        M5.Display.setTextColor(RED);
        M5.Display.println("PSRAM ERR!");
        while(1) delay(100);
    }

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

    M5.Display.setTextColor(GREEN);
    M5.Display.println("Pret!");
    M5.Display.println("");
    M5.Display.println("1. Lancez Audio_1");
    M5.Display.println("2. Appuyez bouton");
    M5.Display.println("   pour enregistrer");
    M5.Display.println("3. Rappuyez pour");
    M5.Display.println("   arreter + envoyer");

    Serial.println("Pret — appuyez bouton pour demarrer");
}

void loop() {
    M5.update();

    // ── DÉMARRER / ARRÊTER enregistrement ────────────────────
    if (M5.BtnA.wasPressed()) {
        if (!recording && !recorded) {
            // Démarrer
            samples_recorded = 0;
            recording = true;
            M5.Display.fillScreen(BLACK);
            M5.Display.setTextColor(RED);
            M5.Display.setTextSize(2);
            M5.Display.setCursor(10, 40);
            M5.Display.println("REC...");
            Serial.println("Enregistrement demarre");

        } else if (recording) {
            // Arrêter
            recording = false;
            recorded  = true;
            Serial.printf("Enregistrement termine: %d samples = %.1fs\n",
                samples_recorded, (float)samples_recorded/SAMPLE_RATE);

            M5.Display.fillScreen(BLACK);
            M5.Display.setTextColor(CYAN);
            M5.Display.setTextSize(1);
            M5.Display.println("Normalisation...");

            // ── Normalisation globale identique a librosa ─────
            int16_t global_max = 0;
            for (int i = 0; i < samples_recorded; i++) {
                int16_t v = abs(audio_buffer[i]);
                if (v > global_max) global_max = v;
            }
            Serial.printf("Max global brut: %d\n", global_max);

            // Normaliser : y = y / max(abs(y)) × 32767
            for (int i = 0; i < samples_recorded; i++) {
                audio_normalized[i] = (int16_t)(
                    (float)audio_buffer[i] / global_max * 32767.0f);
            }

            // Stats après normalisation
            int16_t max_norm = 0;
            int64_t sum_norm = 0;
            for (int i = 0; i < samples_recorded; i++) {
                if (abs(audio_normalized[i]) > max_norm)
                    max_norm = abs(audio_normalized[i]);
                sum_norm += audio_normalized[i];
            }
            float avg_norm = (float)sum_norm / samples_recorded;

            Serial.printf("Max apres normalisation: %d\n", max_norm);
            Serial.printf("Avg apres normalisation: %.2f\n", avg_norm);

            M5.Display.setTextColor(GREEN);
            M5.Display.printf("Max brut   : %d\n", global_max);
            M5.Display.printf("Max normali: %d\n", max_norm);
            M5.Display.printf("Samples    : %d\n", samples_recorded);
            M5.Display.printf("Duree      : %.1fs\n",
                (float)samples_recorded/SAMPLE_RATE);
            M5.Display.println("");
            M5.Display.setTextColor(YELLOW);
            M5.Display.println("Appuyez pour envoyer");
            M5.Display.println("via Serial...");

        } else if (recorded) {
            // Envoyer via Serial
            M5.Display.fillScreen(BLACK);
            M5.Display.setTextColor(CYAN);
            M5.Display.println("Envoi Serial...");
            M5.Display.printf("%d samples\n", samples_recorded);
            M5.Display.println("Ne pas fermer");
            M5.Display.println("le moniteur serie!");

            Serial.println("=== DEBUT_AUDIO ===");
            Serial.printf("SAMPLES:%d\n", samples_recorded);
            Serial.printf("SAMPLERATE:%d\n", SAMPLE_RATE);

            // Envoyer signal brut
            Serial.println("=== BRUT ===");
            for (int i = 0; i < samples_recorded; i++) {
                Serial.println(audio_buffer[i]);
                // Petite pause toutes les 1000 samples
                if (i % 1000 == 0) delay(10);
            }

            // Envoyer signal normalisé
            Serial.println("=== NORMALISE ===");
            for (int i = 0; i < samples_recorded; i++) {
                Serial.println(audio_normalized[i]);
                if (i % 1000 == 0) delay(10);
            }

            Serial.println("=== FIN_AUDIO ===");

            M5.Display.fillScreen(BLACK);
            M5.Display.setTextColor(GREEN);
            M5.Display.println("Envoi termine!");
            M5.Display.println("Recuperez les");
            M5.Display.println("donnees sur PC");
            recorded = false;
        }
    }

    // ── ENREGISTREMENT ────────────────────────────────────────
    if (recording && samples_recorded + 8000 < MAX_SAMPLES) {
        int16_t* dest = audio_buffer + samples_recorded;
        if (M5.Mic.record(dest, 8000, SAMPLE_RATE)) {
            samples_recorded += 8000;

            // Affichage progression
            M5.Display.fillRect(0, 70, 128, 30, BLACK);
            M5.Display.setCursor(0, 70);
            M5.Display.setTextColor(WHITE);
            M5.Display.setTextSize(1);
            M5.Display.printf("%.1fs enregistres\n",
                (float)samples_recorded/SAMPLE_RATE);

            // Barre
            int pct = samples_recorded * 100 / MAX_SAMPLES;
            M5.Display.fillRect(0, 90, pct*128/100, 8, GREEN);
        }

        // PSRAM pleine → arrêt auto
        if (samples_recorded + 8000 >= MAX_SAMPLES) {
            recording = false;
            recorded  = true;
            M5.Display.setTextColor(ORANGE);
            M5.Display.println("PSRAM pleine!");
        }
    }
}