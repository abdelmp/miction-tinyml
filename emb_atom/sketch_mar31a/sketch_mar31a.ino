#include <M5Unified.h>
 
void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
   
    // Activer le moniteur série à 115200 baud
    Serial.begin(115200);
    delay(1000);
    Serial.println("--- DEMARRAGE ACQUISITION AUDIO ---");
 
    M5.Display.setTextSize(2);
    M5.Display.fillScreen(BLACK);
    M5.Display.setTextColor(WHITE);
    M5.Display.println("Init INMP441...");
 
    auto mic_cfg = M5.Mic.config();
    mic_cfg.sample_rate = 16000;
    mic_cfg.pin_bck     = 1;   // SCK -> G1
    mic_cfg.pin_ws      = 2;   // WS  -> G2
    mic_cfg.pin_data_in = 38;  // SD  -> G38
   
    M5.Mic.config(mic_cfg);
   
    if (!M5.Mic.begin()) {
        M5.Display.setTextColor(RED);
        M5.Display.println("Erreur I2S!");
        Serial.println("ERREUR: Impossible d'initialiser le micro I2S");
        while (1) delay(100);
    }
 
    M5.Display.setTextColor(GREEN);
    M5.Display.println("INMP441 OK!");
    Serial.println("INMP441 Initialise avec succes.");
}
 
void loop() {
    static int16_t samples[512];
   
    if (M5.Mic.record(samples, 512, 16000)) {
        float sum_sq = 0;
        for (int i = 0; i < 512; i++) {
            float s = (float)samples[i] / 32768.0f;
            sum_sq += s * s;
        }
        float rms = sqrtf(sum_sq / 512);
 
        // Affichage Ecran
        M5.Display.fillRect(0, 80, 128, 40, BLACK);
        M5.Display.setCursor(0, 80);
        M5.Display.printf("RMS: %.4f\n", rms);
 
        // ENVOI VERS LE MONITEUR SERIE
        Serial.print("RMS:");
        Serial.println(rms, 6); // Affiche 6 décimales
 
        if (rms > 0.01) {
            M5.Display.setTextColor(GREEN);
            M5.Display.println("SON DETECTE");
            Serial.println(">>> EVENEMENT: SON DETECTE");
        } else {
            M5.Display.setTextColor(WHITE);
            M5.Display.println("silence...");
        }
    }
    M5.update();
    delay(50);
}
 