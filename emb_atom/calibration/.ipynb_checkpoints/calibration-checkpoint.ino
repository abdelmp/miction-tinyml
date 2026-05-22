#include <M5Unified.h>

#define SAMPLE_RATE  16000
#define SEG_SAMPLES  8000   // 500ms
#define MAX_SEGMENTS 90     // 90 × 500ms = 45s ← couvre Audio_1 complet

int16_t* segment_buffer = nullptr;
int      seg_stored     = 0;
bool     capturing      = false;
bool     analyse_done   = false;

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    Serial.begin(115200);
    delay(2000);  // attendre USB CDC

    Serial.println("=== VERIFICATION SIGNAL ===");

    // PSRAM
    size_t buf_size = MAX_SEGMENTS * SEG_SAMPLES * sizeof(int16_t);
    segment_buffer  = (int16_t*)heap_caps_malloc(
        buf_size, MALLOC_CAP_SPIRAM);

    if (!segment_buffer) {
        Serial.println("PSRAM ERR!");
        M5.Display.setTextColor(RED);
        M5.Display.println("PSRAM ERR!");
        while(1) delay(100);
    }
    Serial.printf("PSRAM OK — buffer %d KB\n", buf_size/1024);

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

    M5.Display.setTextSize(1);
    M5.Display.fillScreen(BLACK);
    M5.Display.setTextColor(GREEN);
    M5.Display.println("Pret");
    M5.Display.println("Lancez Audio_1");
    M5.Display.println("puis appuyez");

    Serial.println("Pret — appuyez sur le bouton");
}

void loop() {
    M5.update();

    // Démarrer capture
    if (M5.BtnA.wasPressed() && !capturing && !analyse_done) {
        seg_stored = 0;
        capturing  = true;
        Serial.println("\n--- CAPTURE DEMARREE ---");
        Serial.println("Seg | Max   | RMS   | Avg");
        M5.Display.fillScreen(BLACK);
        M5.Display.setTextColor(GREEN);
        M5.Display.println("CAPTURE...");
    }

    // Capture segment par segment
    if (capturing && seg_stored < MAX_SEGMENTS) {
        int16_t* dest = segment_buffer + seg_stored * SEG_SAMPLES;

        if (M5.Mic.record(dest, SEG_SAMPLES, SAMPLE_RATE)) {
            // Calculer stats
            int16_t max_s  = 0;
            int64_t sum_sq = 0;
            int32_t sum    = 0;
            for (int i = 0; i < SEG_SAMPLES; i++) {
                int16_t v = dest[i];
                if (abs(v) > max_s) max_s = abs(v);
                sum_sq += (int64_t)v * v;
                sum    += v;
            }
            float rms = sqrtf((float)sum_sq / SEG_SAMPLES);
            float avg = (float)sum / SEG_SAMPLES;

            seg_stored++;

            // Envoyer via Serial
            Serial.printf("%3d | %5d | %5.0f | %6.1f\n",
                seg_stored, max_s, rms, avg);

            // Afficher sur écran
            M5.Display.fillRect(0, 15, 128, 50, BLACK);
            M5.Display.setCursor(0, 15);
            M5.Display.setTextColor(WHITE);
            M5.Display.printf("Seg %d/%d\n", seg_stored, MAX_SEGMENTS);
            M5.Display.printf("Max:%d\n", max_s);
            M5.Display.printf("RMS:%.0f\n", rms);

            int bar = constrain(map(max_s, 0, 20000, 0, 128), 0, 128);
            M5.Display.fillRect(0, 70, 128, 10, BLACK);
            M5.Display.fillRect(0, 70, bar,  10,
                max_s > 15000 ? RED :
                max_s > 3000  ? GREEN : ORANGE);
        }

        if (seg_stored >= MAX_SEGMENTS) {
            capturing    = false;
            analyse_done = true;
            Serial.println("\n--- CAPTURE TERMINEE ---");
            Serial.printf("Total segments : %d\n", seg_stored);
            Serial.printf("Duree capturee : %.1fs\n", seg_stored * 0.5f);
            M5.Display.fillScreen(BLACK);
            M5.Display.setTextColor(YELLOW);
            M5.Display.println("CAPTURE OK");
            M5.Display.printf("%d segs = %.0fs\n",
                seg_stored, seg_stored * 0.5f);
        }
    }
}