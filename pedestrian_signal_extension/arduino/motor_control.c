/*
 스마트 신호등 - 구동부 시나리오 코드
  NEMA17 + L298N + 풀리 + 서보모터
 
배선
  ENA = 2   IN1 = 3   IN2 = 4
  ENB = 5   IN3 = 6   IN4 = 7
  서보 신호선 = 10
  코일 A(검정,초록) -> OUT1, OUT2
  코일 B(빨강,파랑) -> OUT3, OUT4
 
  1 / 2 / 3 -> 시나리오 실행
  c         -> 캘리브레이션: 정확히 1회전(200스텝)만 이동
 
 */

#include <Servo.h>

// 핀 배치
const int ENA = 2;
const int IN1 = 3;
const int IN2 = 4;
const int ENB = 5;
const int IN3 = 6;
const int IN4 = 7;
const int SERVO_PIN = 10;


const float STEPS_PER_REV = 200.0;

// 실측값으로 보정
const float MM_PER_REV = 76;

const float STEPS_PER_MM = STEPS_PER_REV / MM_PER_REV;

// 시나리오 파라미터
const float DIST_A_MM  = 292.0;   // 시나리오 1,2 (29.2cm)
const float DIST_B1_MM = 245.0;   // 시나리오 3 1구간 (24.5cm)
const float DIST_B2_MM = 50.0;    // 시나리오 3 2구간 (5cm)

const unsigned long T1_MS  = 8000;    // 시나리오 1: 8초
const unsigned long T2_MS  = 14000;   // 시나리오 2: 14초
const unsigned long T3A_MS = 7000;    // 시나리오 3 1구간: 7초

const unsigned long T3B_MS = 1429;
// 복귀
const unsigned int RETURN_DELAY_US = 5000;
const unsigned int RAMP_START_US   = 15000;
const unsigned int RAMP_ACCEL      = 50;

// 서보모터
const int SERVO_HOME   = 0;
const int SERVO_ACTION = 90;
const unsigned long SERVO_SETTLE_MS = 700;

//
const unsigned long TURN_PAUSE_MS = 1000;
const unsigned long WAIT_BEFORE_RETURN_MS = 4000;   // 도착 후 복귀 전 정지 대기
const unsigned long WAIT_AFTER_SERVO_MS   = 5000;   // 쓰러졌을때 대기
const bool RELEASE_WHEN_IDLE = true;   // 정지 중 코일 차단(발열)

// 시퀀스
const uint8_t SEQ[4][4] = {
  {1, 0, 0, 1},
  {1, 0, 1, 0},
  {0, 1, 1, 0},
  {0, 1, 0, 1}
};

Servo actionServo;
int seqIdx = 0;

// ─────────────────────────────────────────────────────
void setup()
{
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  actionServo.attach(SERVO_PIN);
  actionServo.write(SERVO_HOME);

  goIdle();

  Serial.begin(9600);
  Serial.println(F("READY - 1/2/3 = scenario, c = calibrate 1 rev, v = calibrateBack 1 rev"));
}


void enableDriver(bool on)
{
  digitalWrite(ENA, on ? HIGH : LOW);
  digitalWrite(ENB, on ? HIGH : LOW);
}

void applyStep(int idx)
{
  digitalWrite(IN1, SEQ[idx][0]);
  digitalWrite(IN2, SEQ[idx][1]);
  digitalWrite(IN3, SEQ[idx][2]);
  digitalWrite(IN4, SEQ[idx][3]);
}

void releaseCoils()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void goIdle()
{
  if (RELEASE_WHEN_IDLE) {
    releaseCoils();
    enableDriver(false);
  }
}

long mmToSteps(float mm)
{
  return (long)(mm * STEPS_PER_MM + 0.5);
}


void stepDelay(unsigned long us)
{
  if (us >= 16000UL) {
    unsigned long ms = us / 1000UL;
    delay(ms);
    delayMicroseconds((unsigned int)(us - ms * 1000UL));
  } else {
    delayMicroseconds((unsigned int)us);
  }
}

// 캘리브레이션 이동
void stepMove(long steps, bool forward, unsigned int delayUs)
{
  enableDriver(true);
  for (long s = 0; s < steps; s++) {
    seqIdx = forward ? ((seqIdx + 1) & 3) : ((seqIdx + 3) & 3);
    applyStep(seqIdx);
    delayMicroseconds(delayUs);
  }
}

// 시나리오 전진
void moveConstant(float mm, bool forward, unsigned long durationMs)
{
  long steps = mmToSteps(mm);
  if (steps <= 0) return;

  unsigned long d = (durationMs * 1000UL) / (unsigned long)steps;

  enableDriver(true);
  for (long s = 0; s < steps; s++) {
    seqIdx = forward ? ((seqIdx + 1) & 3) : ((seqIdx + 3) & 3);
    applyStep(seqIdx);
    stepDelay(d);
  }
}

// 복귀
void moveRamped(float mm, bool forward, unsigned int minDelayUs)
{
  long steps = mmToSteps(mm);
  if (steps <= 0) return;

  enableDriver(true);

  unsigned int d = RAMP_START_US;
  long accelSteps = (long)((RAMP_START_US - minDelayUs) / RAMP_ACCEL);
  if (accelSteps > steps / 2) accelSteps = steps / 2;
  long decelStart = steps - accelSteps;

  for (long s = 0; s < steps; s++) {
    seqIdx = forward ? ((seqIdx + 1) & 3) : ((seqIdx + 3) & 3);
    applyStep(seqIdx);
    delayMicroseconds(d);

    if (s < decelStart) {
      if (d > minDelayUs) {
        d -= RAMP_ACCEL;
        if (d < minDelayUs) d = minDelayUs;
      }
    } else {
      if (d < RAMP_START_US) d += RAMP_ACCEL;
    }
  }
}

// 1회전 캘리브레이션
void calibrate()
{
  Serial.println(F("CAL: 200 steps = 1 revolution. Measure the distance."));
  stepMove((long)STEPS_PER_REV, true, 8000);
  goIdle();
  Serial.print(F("CAL done."));
}

// 캘리브레이션 복귀
void calibrateBack()
{
  Serial.println(F("CAL: reverse 200 steps"));
  stepMove((long)STEPS_PER_REV, false, 8000);
  goIdle();
  Serial.println(F("CAL back done"));
}

// 시나리오 1,2,3
void scenario1()
{
  Serial.println(F("S1: 292mm / 8s (normal)"));
  moveConstant(DIST_A_MM, true, T1_MS);
  goIdle();
  delay(TURN_PAUSE_MS);

  delay(WAIT_BEFORE_RETURN_MS);

  moveRamped(DIST_A_MM, false, RETURN_DELAY_US);
  goIdle();
  Serial.println(F("S1 done"));
}

void scenario2()
{
  Serial.println(F("S2: 292mm / 14s (slow)"));
  moveConstant(DIST_A_MM, true, T2_MS);
  goIdle();
  delay(TURN_PAUSE_MS);

  delay(WAIT_BEFORE_RETURN_MS);

  moveRamped(DIST_A_MM, false, RETURN_DELAY_US);
  goIdle();
  Serial.println(F("S2 done"));
}

void scenario3()
{
  Serial.println(F("S3: 245mm / 7s"));
  moveConstant(DIST_B1_MM, true, T3A_MS);
  goIdle();
  delay(TURN_PAUSE_MS);

  Serial.println(F("S3: servo -> 90"));
  actionServo.write(SERVO_ACTION);
  delay(SERVO_SETTLE_MS);
  delay(WAIT_AFTER_SERVO_MS);

  Serial.println(F("S3: 50mm"));
  moveConstant(DIST_B2_MM, true, T3B_MS);
  goIdle();
  delay(TURN_PAUSE_MS);

  Serial.println(F("S3: servo -> home"));
  actionServo.write(SERVO_HOME);
  delay(SERVO_SETTLE_MS);

  delay(WAIT_BEFORE_RETURN_MS);

  Serial.println(F("S3: return 295mm"));
  moveRamped(DIST_B1_MM + DIST_B2_MM, false, RETURN_DELAY_US);
  goIdle();
  Serial.println(F("S3 done"));
}


void loop()
{
  if (Serial.available()) {
    char ch = Serial.read();
    switch (ch) {
      case '1': scenario1(); break;
      case '2': scenario2(); break;
      case '3': scenario3(); break;
      case 'c':
      case 'C': calibrate(); break;
      case 'v':
      case 'V': calibrateBack(); break;
      default:  break;
    }
  }
}
