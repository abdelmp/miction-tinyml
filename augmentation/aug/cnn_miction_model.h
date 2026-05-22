// CNN 2D Miction — Header pour ESP32-S3
#ifndef CNN_MICTION_MODEL_H
#define CNN_MICTION_MODEL_H

#include <stdint.h>

extern const unsigned char cnn_miction_model[];
extern const int cnn_miction_model_len;

// Paramètres du modèle
#define MODEL_INPUT_H     64    // Hauteur spectrogramme Mel
#define MODEL_INPUT_W     51    // Largeur (frames temporelles)
#define MODEL_INPUT_C      1    // Canaux (mono)
#define MODEL_N_CLASSES    4    // Nombre de classes
#define MODEL_SEGMENT_MS 500    // Durée segment en ms
#define MODEL_HOP_MS     250    // Hop chevauchement en ms
#define MODEL_SR       16000    // Fréquence échantillonnage
#define MODEL_N_MELS      64    // Bandes Mel
#define MODEL_N_FFT      512    // Taille FFT
#define MODEL_HOP_LEN    160    // Hop longueur Mel

// Classes (ordre LabelEncoder)
// 0 = bruit_ambiant
// 1 = bruit_chasse
// 2 = miction_active
// 3 = silence

#endif // CNN_MICTION_MODEL_H
