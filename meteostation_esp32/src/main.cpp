#include <Arduino.h>
#include <SHT2x.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <mbedtls/md.h>
#include <base64.hpp>

SHT2x sht;
const int adcPin = 33;
float factorTensiune = 2.0;

const int anemPin = 32;
const int rainfallPin = 25;  // Tipping bucket rain gauge
volatile unsigned long rotatii = 0;
volatile unsigned long tipsBucket = 0;
volatile unsigned long lastTipTime = 0;
const unsigned long DEBOUNCE_MS = 100;  // Debounce for rain gauge switch
const float MM_PER_TIP = 0.2794;  // mm of rain per bucket tip
unsigned long ultimaMasura = 0;

const char* ssid = "iiap5_2g";
const char* passwd = "ionutqwerty";

// JWT Secret - MUST match your Flask server's SECRET_KEY
const char* JWT_SECRET = "V3fryS3cr3tK3y,7h47Y0uC4n7F1nd:)!";
// Wind direction lookup table based on measured voltages
// Each entry: {voltage, direction name, degrees}
struct WindDirection {
    float voltage;
    const char* cardinal;
    float degrees;
};

// Sorted by voltage for easier matching
const WindDirection directions[] = {
    {0.00, "NE",  45.0},   // NE alternate (two switches)
    {0.14, "NW", 315.0},   // NW alternate (two switches)
    {0.18, "N",    0.0},   // North
    {0.30, "E",   90.0},   // East alternate (two switches)
    {0.50, "NE",  45.0},   // NE primary
    {1.00, "E",   90.0},   // East primary
    {1.50, "NW", 315.0},   // NW primary
    {2.00, "SE", 135.0},   // SE
    {2.50, "W",  270.0},   // West
    {2.60, "SW", 225.0},   // SW alternate
    {2.70, "SW", 225.0},   // SW primary
    {2.80, "S",  180.0},   // South
};
const int numDirections = sizeof(directions) / sizeof(directions[0]);

const char* getWindDirection(float voltage, float* degrees) {
    float minDiff = 999.0;
    int bestMatch = 0;
    
    for (int i = 0; i < numDirections; i++) {
        float diff = abs(voltage - directions[i].voltage);
        if (diff < minDiff) {
            minDiff = diff;
            bestMatch = i;
        }
    }
    
    *degrees = directions[bestMatch].degrees;
    return directions[bestMatch].cardinal;
}
void IRAM_ATTR isrAnemometru() {
  rotatii++;
}

void IRAM_ATTR isrRainfall() {
  unsigned long now = millis();
  if (now - lastTipTime > DEBOUNCE_MS) {
    tipsBucket++;
    lastTipTime = now;
  }
}

struct TemperatureHumidity {
    float temperature;
    float humidity;
};
TemperatureHumidity readSHT()
{
    if (sht.read())
    {
        float temp = sht.getTemperature();
        float hum = sht.getHumidity();
        return {temp, hum};
    }
    return {-500.0, -500.0}; 
}
float vout, vin;
uint16_t raw;
void readVoltage()
{
    raw = analogRead(adcPin);
    vout = (raw / 4095.0) * 3.3;
    vin = vout * factorTensiune;
}
struct MeteoData {
    float windSpeed;
    float windDirectionDegrees;
    float windDirectionVoltage;
    float rainfall;
    const char* windDirectionCardinal;
};
MeteoData readWindSpeed()
{
    unsigned long rotatiiSec = rotatii;
    rotatii = 0;
    
    // Capture and reset rainfall tips
    unsigned long tips = tipsBucket;
    tipsBucket = 0;
    float rainfall = tips * MM_PER_TIP;

    float vitezaKmH = (rotatiiSec * 2.4) / 2.3;

    ultimaMasura = millis();
    float degrees;
    const char* cardinal = getWindDirection(vin, &degrees);
    
    return {vitezaKmH, degrees, vin, rainfall, cardinal};
    // Serial.printf("Directie vant: %s (%.1f°) [%.2fV] | Viteza: %.1f km/h\n",
    //               cardinal, degrees, vin, vitezaKmH);
}
String urlEncode(const char* str) {
    String encoded = "";
    char c;
    while ((c = *str++)) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            encoded += c;
        } else {
            char buf[4];
            sprintf(buf, "%%%02X", (unsigned char)c);
            encoded += buf;
        }
    }
    return encoded;
}

// ==================== JWT GENERATION ====================

String base64UrlEncode(const unsigned char* data, size_t len) {
    // Calculate base64 encoded length
    size_t encodedLen = encode_base64_length(len);
    unsigned char* encoded = new unsigned char[encodedLen + 1];
    
    // Encode to base64
    encode_base64(data, len, encoded);
    
    String result = String((char*)encoded);
    delete[] encoded;
    
    // Convert to base64url: replace + with -, / with _, remove =
    result.replace('+', '-');
    result.replace('/', '_');
    while (result.endsWith("=")) {
        result.remove(result.length() - 1);
    }
    
    return result;
}

String base64UrlEncodeString(const String& str) {
    return base64UrlEncode((const unsigned char*)str.c_str(), str.length());
}

String getChipIdString() {
    uint64_t chipId = ESP.getEfuseMac();
    char chipIdStr[21];
    snprintf(chipIdStr, sizeof(chipIdStr), "%llu", chipId);
    return String(chipIdStr);
}

String generateJWT(const char* chipId) {
    // Header: {"alg":"HS256","typ":"JWT"}
    String header = "{\"alg\":\"HS256\",\"typ\":\"JWT\"}";
    String headerB64 = base64UrlEncodeString(header);
    
    // Payload with chip_id and a far-future validity
    // Server will handle actual validation, this just needs valid format
    String payload = String("{\"chip_id\":\"") + chipId + "\",\"valability\":\"2099-12-31T23:59:59.000000+00:00\"}";
    String payloadB64 = base64UrlEncodeString(payload);
    
    // Message to sign: header.payload
    String message = headerB64 + "." + payloadB64;
    
    // HMAC-SHA256 signature
    unsigned char hmac[32];
    mbedtls_md_context_t ctx;
    mbedtls_md_init(&ctx);
    mbedtls_md_setup(&ctx, mbedtls_md_info_from_type(MBEDTLS_MD_SHA256), 1);
    mbedtls_md_hmac_starts(&ctx, (const unsigned char*)JWT_SECRET, strlen(JWT_SECRET));
    mbedtls_md_hmac_update(&ctx, (const unsigned char*)message.c_str(), message.length());
    mbedtls_md_hmac_finish(&ctx, hmac);
    mbedtls_md_free(&ctx);
    
    String signatureB64 = base64UrlEncode(hmac, 32);
    
    // Complete JWT: header.payload.signature
    return message + "." + signatureB64;
}

int send_segment(const char* url, HTTPClient& http, WiFiClientSecure& client, TemperatureHumidity th, MeteoData md, const char* jwt, String& jsonPayload){
    // Build URL with JWT as query parameter (URL encoded)
    String fullUrl = String(url) + "?jwt=" + urlEncode(jwt);
    http.begin(client, fullUrl);
    http.addHeader("Content-Type", "application/json");
    readVoltage();
    jsonPayload = String("{") +
        "\"temperature\":" + String(th.temperature, 2) + "," +
        "\"humidity\":" + String(th.humidity, 2) + "," +
        "\"wind_speed\":" + String(md.windSpeed, 2) + "," +
        "\"rainfall\":" + String(md.rainfall, 1) + "," +
        "\"wind_direction_degrees\":" + String(md.windDirectionDegrees, 2) + "," +
        "\"wind_direction_voltage\":" + String(md.windDirectionVoltage, 2) + "," +
        "\"ssid\":\"" + ssid + "\"" +
    "}";
    // Send as GET with body
    int httpResponseCode = http.POST(jsonPayload);
    return httpResponseCode;
}
// JWT token storage - will be generated in setup() with real chip_id
char jwtToken[512] = "";
char* header = jwtToken;

const char* send_data = "https://192.168.0.177:5500/get_data";
const char* generate_token = "https://192.168.0.177:5500/generate_token";
const char* request_access = "https://192.168.0.177:5500/request";
const char* current_url = send_data;

enum CURRENT_STATES{
    SEND_STATE,
    GENERATE_TOKEN_STATE,
    REQUEST_ACCESS_STATE
}CURRENT_STATE;

void setup()
{
    Serial.begin(115200);
    Wire.begin();
    sht.begin();
    WiFi.begin(ssid, passwd);
    while (WiFi.status() != WL_CONNECTED){
        delay(500);
        Serial.print(".");
    }
    if(WiFi.status() == WL_CONNECTED){
        Serial.println("\nWiFi connected " + WiFi.localIP().toString());
        
        // Generate JWT with real chip_id
        String chipId = getChipIdString();
        String jwt = generateJWT(chipId.c_str());
        strncpy(jwtToken, jwt.c_str(), sizeof(jwtToken) - 1);
        jwtToken[sizeof(jwtToken) - 1] = '\0';
        Serial.println("Generated JWT DONE ");
    }else {
        Serial.println("\nWiFi FAILED! Status: ");
        switch(WiFi.status()) {
            case WL_NO_SSID_AVAIL: Serial.println("SSID not found"); break;
            case WL_CONNECT_FAILED: Serial.println("Wrong password"); break;
            case WL_DISCONNECTED: Serial.println("Disconnected"); break;
            default: Serial.printf("Code: %d\n", WiFi.status()); break;
        }
        Serial.println("Restarting in 5 seconds...");
        delay(5000);
        ESP.restart();
    }
    CURRENT_STATE = SEND_STATE;
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
    
    pinMode(anemPin, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(anemPin), isrAnemometru, FALLING);
    
    // Rain gauge - tipping bucket (0.2794mm per tip)
    pinMode(rainfallPin, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(rainfallPin), isrRainfall, FALLING);
}

void loop()
{
    WiFiClientSecure client;
    client.setInsecure();  // Skip SSL certificate verification (for self-signed certs)
    
    HTTPClient http;
    http.setTimeout(10000);  // 10 second timeout
    
    TemperatureHumidity th = readSHT();
    MeteoData md = readWindSpeed();
    String jsonPayload;
    String response;
    int code = send_segment(current_url, http, client, th, md, header, jsonPayload);
    switch(CURRENT_STATE){
        case SEND_STATE:
            if(code == 200){
                response = http.getString();
                Serial.println("Data sent successfully: " + response);
            } if(code == 405){
                Serial.println("Generating new token...");
                CURRENT_STATE = GENERATE_TOKEN_STATE;
                current_url = generate_token;
            } if(code == 403){
                response = http.getString();
                Serial.println(response);
            } if(code == 400){
                // Serial.println("Bad Request. Check the sent data.");
                response = http.getString();
                Serial.println(response + "\nPayload was: " + jsonPayload);
            }
            break;
        case GENERATE_TOKEN_STATE:
            if(code == 200){
                response = http.getString();
                header = strdup(response.c_str());
                CURRENT_STATE = SEND_STATE;
                current_url = send_data;
            } if(code == 403){
                response = http.getString();
                Serial.println(response);
                CURRENT_STATE = REQUEST_ACCESS_STATE;
                current_url = request_access;
            } if(code == 405){
                response = http.getString();
                Serial.println(response);
            }
            break;
        case REQUEST_ACCESS_STATE:
            if(code == 200){
                response = http.getString();
                Serial.println(response);
                CURRENT_STATE = GENERATE_TOKEN_STATE;
                current_url = generate_token;
            }if(code == 400){
                response = http.getString();
                Serial.println(response);
            }if(code == 405){
                response = http.getString();
                Serial.println(response);
            }
            break;
    }
    http.end();
    delay(5000);
}
