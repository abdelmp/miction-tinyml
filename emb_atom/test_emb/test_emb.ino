#include <M5Unified.h>

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    M5.Display.setTextSize(1);
    M5.Display.fillScreen(BLACK);
    M5.Display.setTextColor(WHITE);
    M5.Display.println("Test micro...");

    auto mic_cfg = M5.Mic.config();
    mic_cfg.sample_rate = 16000;
    mic_cfg.pin_bck     = 1;   // ← tester d'abord avec 1,2,38
    mic_cfg.pin_ws      = 2;
    mic_cfg.pin_data_in = 38;
    mic_cfg.magnification = 1;  // ← ajouter cette ligne — gain ×16
    M5.Mic.config(mic_cfg);

    if (!M5.Mic.begin()) {
        M5.Display.setTextColor(RED);
        M5.Display.println("ERREUR micro!");
        return;
    }
    M5.Display.setTextColor(GREEN);
    M5.Display.println("Micro OK");
}

void loop() {
    static int16_t samples[512];
    if (M5.Mic.record(samples, 512, 16000)) {
        int16_t max_s = 0;
        for (int i = 0; i < 512; i++) {
            int16_t v = abs(samples[i]);
            if (v > max_s) max_s = v;
        }
        M5.Display.fillRect(0, 40, 128, 20, BLACK);
        M5.Display.setCursor(0, 40);
        M5.Display.printf("Max: %d", max_s);

        // Barre de volume
        M5.Display.fillRect(0, 70, 128, 15, BLACK);
        int bar = map(max_s, 0, 10000, 0, 128);
        bar = constrain(bar, 0, 128);
        M5.Display.fillRect(0, 70, bar, 15,
            max_s > 500 ? GREEN : RED);
    }
    M5.update();
}