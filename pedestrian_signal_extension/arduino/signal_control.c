/*
  스마트 신호등 제어 (Arduino Uno) 

  세그먼트: 5621BS (0.56" 2자리 공통 애노드, 10핀)
  실측 확정된 물리핀 - 아두이노 핀 - 실제 획:
    물리1 -> D12 -> c
    물리2 -> (미사용, dp)
    물리3 -> D11 -> e   (물리핀 2에서 옮겨온 자리)
    물리4 -> D10 -> d
    물리5 -> A0  -> g
    물리6 -> D9  -> f
    물리9 -> D13 -> b
    물리10-> D8  -> a
    물리7/8 -> A1/A2 (공통, 기존과 동일)

  즉 D8~A0가 표준 a,f,d,e,c,b,g 순서로 꽂혀 있음.
   배선 (2N3906 PNP 트랜지스터 2개 추가 버전 - 밝기 개선)
    D2 차량빨강  D3 차량노랑  D4 차량초록  D5 보행빨강  D6 보행초록  (각 220Ω)
    D7 부저
    D8=a  D9=f  D10=d  D11=e  D12=c  D13=b  A0=g   (실측 순서, 각 220Ω 추가)
    A1 --[1kΩ]-- Q1(2N3906) 베이스, 이미터-5V, 컬렉터-물리핀8(10의 자리 공통)
    A2 --[1kΩ]-- Q2(2N3906) 베이스, 이미터-5V, 컬렉터-물리핀7(1의 자리 공통)

  파이 명령: normal / zone1~zone5 / fall
  Serial 115200 (파이 쪽도 동일하게 맞출 것)
*/

const int VEHICLE_RED    = 2;
const int VEHICLE_YELLOW = 3;
const int VEHICLE_GREEN  = 4;
const int PED_RED        = 5;
const int PED_GREEN      = 6;
const int BUZZER_PIN     = 7;

const int SEG_PINS[7] = {8, 9, 10, 11, 12, 13, A0};  // 실제 배선 순서: a f d e c b g
const uint8_t SEG_BIT_FOR_PIN[7] = {0, 5, 3, 4, 2, 1, 6}; // D8=a(0) D9=f(5) D10=d(3) D11=e(4) D12=c(2) D13=b(1) A0=g(6)
const int DIGIT_TENS = A1;
const int DIGIT_ONES = A2;
const int DIGIT_ON = HIGH;
const int DIGIT_OFF = LOW;

const unsigned long DUR_VEHICLE_GREEN  = 20000;
const unsigned long DUR_VEHICLE_YELLOW = 2000;
const unsigned long DUR_PED_GREEN      = 10000;

const int EXTEND_CAP_SEC = 10;
const int ZONE_MAX_EXTEND_SEC = 2;
const int ZONE_IDEAL_REMAINING[5] = {10, 8, 6, 4, 2}; 

enum LightPhase { PHASE_VEHICLE_GREEN, PHASE_VEHICLE_YELLOW, PHASE_PED_GREEN };
LightPhase currentPhase = PHASE_VEHICLE_GREEN;
unsigned long phaseStartTime = 0;

enum SystemMode { MODE_NORMAL, MODE_FALL };
SystemMode currentMode = MODE_NORMAL;

const unsigned long BEEP_MS = 300;
bool fallBeepState = false;
unsigned long fallLastToggle = 0;

unsigned long pedGreenDuration = DUR_PED_GREEN;
int extensionUsedSec = 0;

//zone 신호 중복 방지: 직전에 처리한 zone 번호
int lastZoneNum = -1;

const byte DIGIT_PATTERN[10] = {
  0x3F, 0x06, 0x5B, 0x4F, 0x66,
  0x6D, 0x7D, 0x07, 0x7F, 0x6F
};

int displayValue = -1;
uint8_t activeDigit = 0;
unsigned long lastDigitSwitch = 0;
unsigned long currentSlotMs = 4;

void setDisplayNumber(int n) { displayValue = n; }
void clearSegment() { displayValue = -1; }

void allDigitsOff() {
  digitalWrite(DIGIT_TENS, DIGIT_OFF);
  digitalWrite(DIGIT_ONES, DIGIT_OFF);
}

uint8_t litCount(byte p) {
  uint8_t n = 0;
  for (uint8_t i = 0; i < 7; i++) if (p & (1 << i)) n++;
  return n;
}

void writeSegmentPattern(byte pattern) {
  for (uint8_t i = 0; i < 7; i++) {
    bool on = pattern & (1 << SEG_BIT_FOR_PIN[i]);
    digitalWrite(SEG_PINS[i], on ? LOW : HIGH);  
  }
}

void refreshDisplay() {
  if (displayValue >= 0 && displayValue < 10) {
    digitalWrite(DIGIT_TENS, DIGIT_OFF);
    writeSegmentPattern(DIGIT_PATTERN[displayValue % 10]);
    digitalWrite(DIGIT_ONES, DIGIT_ON);
    return;
  }

  if (displayValue < 0) {
    allDigitsOff();
    return;
  }

  if (millis() - lastDigitSwitch < currentSlotMs) return;
  lastDigitSwitch = millis();

  allDigitsOff();
  activeDigit ^= 1;

  byte pattern = (activeDigit == 0)
    ? DIGIT_PATTERN[(displayValue / 10) % 10]
    : DIGIT_PATTERN[displayValue % 10];

  uint8_t n = litCount(pattern);
  currentSlotMs = (n < 2) ? 2 : n; 

  writeSegmentPattern(pattern);

  if (pattern != 0x00) {
    digitalWrite((activeDigit == 0) ? DIGIT_TENS : DIGIT_ONES, DIGIT_ON);
  }
}


void setLights(bool vR, bool vY, bool vG, bool pR, bool pG) {
  digitalWrite(VEHICLE_RED, vR);
  digitalWrite(VEHICLE_YELLOW, vY);
  digitalWrite(VEHICLE_GREEN, vG);
  digitalWrite(PED_RED, pR);
  digitalWrite(PED_GREEN, pG);
}

void enterPhase(LightPhase phase) {
  currentPhase = phase;
  phaseStartTime = millis();

  switch (phase) {
    case PHASE_VEHICLE_GREEN:
      setLights(false, false, true, true, false);
      clearSegment(); 
      break;
    case PHASE_VEHICLE_YELLOW:
      setLights(false, true, false, true, false);
      clearSegment();
      break;
    case PHASE_PED_GREEN:
      setLights(true, false, false, false, true);
      pedGreenDuration = DUR_PED_GREEN;
      extensionUsedSec = 0;
      lastZoneNum = -1; 
      break;
  }
}

void enterFallState() {
  currentMode = MODE_FALL;
  setLights(true, false, false, true, false);
  clearSegment();
  fallBeepState = false;
  fallLastToggle = millis();
  digitalWrite(BUZZER_PIN, LOW);
}

void updateFallState() {
  if (millis() - fallLastToggle >= BEEP_MS) {
    fallLastToggle = millis();
    fallBeepState = !fallBeepState;
    digitalWrite(BUZZER_PIN, fallBeepState ? HIGH : LOW);
  }
}

void exitFallState() {
  currentMode = MODE_NORMAL;
  digitalWrite(BUZZER_PIN, LOW);
  enterPhase(PHASE_VEHICLE_GREEN);
}

int remainingSecCeil(unsigned long duration, unsigned long elapsed) {
  return (int)((duration - elapsed + 999UL) / 1000UL);
}

void updateNormalCycle() {
  unsigned long elapsed = millis() - phaseStartTime;

  switch (currentPhase) {
    case PHASE_VEHICLE_GREEN:
      if (elapsed >= DUR_VEHICLE_GREEN) { enterPhase(PHASE_VEHICLE_YELLOW); break; }
      break;

    case PHASE_VEHICLE_YELLOW:
      if (elapsed >= DUR_VEHICLE_YELLOW) enterPhase(PHASE_PED_GREEN);
      break;

    case PHASE_PED_GREEN:
      if (elapsed >= pedGreenDuration) { enterPhase(PHASE_VEHICLE_GREEN); break; }
      setDisplayNumber(remainingSecCeil(pedGreenDuration, elapsed));
      break;
  }
}

long getPedGreenRemainingSec() {
  unsigned long elapsed = millis() - phaseStartTime;
  if (elapsed >= pedGreenDuration) return 0;
  return remainingSecCeil(pedGreenDuration, elapsed);
}

char buf[32];
uint8_t idx = 0;

void handleCommand(const char* cmd) {
  if (strcmp(cmd, "normal") == 0) {
    if (currentMode == MODE_FALL) {
      exitFallState();
    }
    Serial.println("OK n");

  } else if (strncmp(cmd, "zone", 4) == 0) {
    int zoneNum = atoi(cmd + 4);

    if (zoneNum < 1 || zoneNum > 5) {
      Serial.println("ERR z");
    } else if (currentMode != MODE_NORMAL || currentPhase != PHASE_PED_GREEN) {
      Serial.println("IGN");
    } else if (zoneNum == lastZoneNum) {
      Serial.println("DUP");
    } else {
      lastZoneNum = zoneNum; 

      long deficit = ZONE_IDEAL_REMAINING[zoneNum - 1] - getPedGreenRemainingSec();

      if (deficit <= 0) {
        Serial.println("OK 0");
      } else {
        int a = (deficit < ZONE_MAX_EXTEND_SEC) ? (int)deficit : ZONE_MAX_EXTEND_SEC;
        int cap = EXTEND_CAP_SEC - extensionUsedSec;
        int allowed = (a < cap) ? a : cap;

        if (allowed <= 0) {
          Serial.println("OK cap");
        } else {
          pedGreenDuration += (unsigned long)allowed * 1000;
          extensionUsedSec += allowed;
          Serial.print("OK +");
          Serial.println(allowed);
        }
      }
    }

  } else if (strcmp(cmd, "fall") == 0) {
    enterFallState();
    Serial.println("OK f");

  } else {
    Serial.println("ERR");
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(VEHICLE_RED, OUTPUT);
  pinMode(VEHICLE_YELLOW, OUTPUT);
  pinMode(VEHICLE_GREEN, OUTPUT);
  pinMode(PED_RED, OUTPUT);
  pinMode(PED_GREEN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  for (uint8_t i = 0; i < 7; i++) pinMode(SEG_PINS[i], OUTPUT);
  pinMode(DIGIT_TENS, OUTPUT);
  pinMode(DIGIT_ONES, OUTPUT);
  allDigitsOff();

  digitalWrite(BUZZER_PIN, LOW);
  enterPhase(PHASE_VEHICLE_GREEN);
  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (idx > 0) { buf[idx] = '\0'; handleCommand(buf); idx = 0; }
    } else if (idx < sizeof(buf) - 1) {
      buf[idx++] = c;
    }
  }

  if (currentMode == MODE_NORMAL) updateNormalCycle();
  else updateFallState();

  refreshDisplay();
}