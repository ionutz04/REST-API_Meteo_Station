#include <Arduino.h>


const int adcPin = 33;  // Schimbat de la 36 la 33
float factorTensiune = 2.0;

// Anemometru pe GPIO32 + control releu
const int anemPin = 32;  // Schimbat de la 33 la 32
const int releuPin = 25;
volatile unsigned long rotatii = 0;
unsigned long ultimaMasura = 0;
bool circuitInchis = false;

void IRAM_ATTR isrAnemometru() {
  rotatii++;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  
  // Anemometru pe GPIO32
  pinMode(anemPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(anemPin), isrAnemometru, FALLING);
  
  // Releu pe GPIO25
  pinMode(releuPin, OUTPUT);
  digitalWrite(releuPin, HIGH);  // Deschide circuitul inițial
  
  ultimaMasura = millis();
}

void loop() {
  // 1. Măsurare tensiune pe GPIO33 (divizor 10k+10k)
  int raw = analogRead(adcPin);
  float vout = (raw / 4095.0) * 3.3;
  float vin = vout * factorTensiune;
  
  // 2. Calcul viteză vânt la 1 secundă
  if (millis() - ultimaMasura >= 1000) {
    unsigned long rotatiiSec = rotatii;
    rotatii = 0;
    
    // Calibrare: 2.3 rotații/sec = 2.4 km/h
    float vitezaKmH = (rotatiiSec * 2.4) / 2.3;
    
    ultimaMasura = millis();
    
    // Afișare pe Serial Monitor
    Serial.printf("Tensiune: %.2fV | Viteza: %.1f km/h \n", 
                  vin, vitezaKmH);
  }
  
  delay(100);
}