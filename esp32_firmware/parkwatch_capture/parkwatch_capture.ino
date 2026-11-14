/*
 * ParkWatch — capture firmware (ESP32-CAM -> Pi)
 *
 * The ESP32 knows nothing about schedules or light: it captures non-stop every
 * CAPTURE_INTERVAL_MS and POSTs the JPEG to a local HTTP endpoint on the
 * Raspberry Pi. All the day/night filtering lives on the Pi
 * (pi_receiver/app.py, using astral).
 *
 * Tested board: Freenove ESP32-S3-WROOM CAM (16MB flash, 8MB OPI PSRAM).
 * Pinout = CAMERA_MODEL_ESP32S3_EYE (confirmed in earlier field tests).
 *
 * Before compiling:
 *   - Arduino IDE: PSRAM = "OPI PSRAM", USB CDC On Boot = Disabled
 *   - Create parkwatch_secrets.h (see parkwatch_secrets.h.example) with your
 *     SSID/password and the Pi's IP. It is not committed (see .gitignore).
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include "parkwatch_secrets.h"

// --- OV5640 pinout on the Freenove ESP32-S3-WROOM CAM (= ESP32S3_EYE) -------
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5
#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13

// --- Board LEDs (the red power LED is wired in hardware and cannot be turned off)
#define LED_GPIO_NUM      2    // blue user LED
#define RGB_GPIO_NUM      48   // WS2812 RGB

// --- Capture parameters -----------------------------------------------------
static const unsigned long CAPTURE_INTERVAL_MS = 3UL * 60UL * 1000UL;  // 3 min
static const framesize_t FRAME_SIZE = FRAMESIZE_UXGA;                  // 1600x1200
static const int JPEG_QUALITY = 12;                                    // 0-63, lower = better quality
static const int HTTP_TIMEOUT_MS = 10000;
static const int MAX_UPLOAD_RETRIES = 3;

unsigned long lastCaptureMs = 0;

// Diagnostics server: GET /capture returns a JPEG right away (bypasses the Pi)
WebServer server(80);

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 9000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAME_SIZE;
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[cam] camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

void connectWifi() {
  Serial.printf("[wifi] connecting to %s...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (millis() - start > 20000) {
      Serial.println("\n[wifi] timeout, retrying begin()");
      WiFi.disconnect(true);
      delay(200);
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      start = millis();
    }
  }
  Serial.printf("\n[wifi] connected, IP=%s\n", WiFi.localIP().toString().c_str());
}

bool uploadFrame(camera_fb_t *fb) {
  HTTPClient http;
  String url = String("http://") + PI_HOST + ":" + PI_PORT + "/upload";
  http.begin(url);
  http.addHeader("Content-Type", "image/jpeg");
  http.setTimeout(HTTP_TIMEOUT_MS);

  int code = http.POST(fb->buf, fb->len);
  bool ok = (code == 200);
  if (ok) {
    Serial.printf("[upload] OK (%d bytes) -> %s\n", fb->len, http.getString().c_str());
  } else {
    Serial.printf("[upload] failed, HTTP %d: %s\n", code, http.errorToString(code).c_str());
  }
  http.end();
  return ok;
}

void captureAndSend() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[cam] fb_get returned NULL, skipping this cycle");
    return;
  }

  bool sent = false;
  for (int attempt = 1; attempt <= MAX_UPLOAD_RETRIES && !sent; attempt++) {
    if (WiFi.status() != WL_CONNECTED) {
      connectWifi();
    }
    sent = uploadFrame(fb);
    if (!sent && attempt < MAX_UPLOAD_RETRIES) {
      delay(1000UL * attempt);  // simple backoff
    }
  }
  if (!sent) {
    Serial.println("[upload] retries exhausted, photo lost (Pi/network down)");
  }

  esp_camera_fb_return(fb);
}

void handleCapture() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "fb_get returned NULL");
    return;
  }
  server.sendHeader("Content-Disposition", "inline; filename=capture.jpg");
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  Serial.printf("[http] /capture served (%d bytes)\n", fb->len);
  esp_camera_fb_return(fb);
}

void ledsOff() {
  pinMode(LED_GPIO_NUM, OUTPUT);
  digitalWrite(LED_GPIO_NUM, LOW);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  rgbLedWrite(RGB_GPIO_NUM, 0, 0, 0);
#else
  neopixelWrite(RGB_GPIO_NUM, 0, 0, 0);
#endif
}

void setup() {
  ledsOff();
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[boot] ParkWatch capture firmware");

  if (!initCamera()) {
    Serial.println("[boot] no camera, restarting in 5s...");
    delay(5000);
    ESP.restart();
  }
  connectWifi();

  server.on("/capture", HTTP_GET, handleCapture);
  server.begin();
  Serial.printf("[http] diagnostics at http://%s/capture\n", WiFi.localIP().toString().c_str());

  // First capture right after boot, then every CAPTURE_INTERVAL_MS
  lastCaptureMs = millis() - CAPTURE_INTERVAL_MS;
}

void loop() {
  server.handleClient();
  if (millis() - lastCaptureMs >= CAPTURE_INTERVAL_MS) {
    lastCaptureMs = millis();
    captureAndSend();
  }
  delay(10);
}
