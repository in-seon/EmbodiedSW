/*
 * [진단용] 부저 단독 테스트 — 시리얼도 파이도 쓰지 않는다.
 *
 * 목적: "부저가 안 운다"의 원인이 통신인지 부저/배선인지 가른다.
 *
 *   이 스케치로 소리가 남   -> 부저·배선은 정상. 문제는 통신 쪽이다.
 *   이 스케치로도 무음      -> 부저·배선 문제. 아래 A/B 중 어느 구간에서 울렸는지 본다.
 *
 * 동작: 두 방식을 번갈아 시도한다. 보드의 LED(13번)로 지금 어느 구간인지 알 수 있다.
 *
 *   구간 A (LED 켜짐) : digitalWrite HIGH/LOW   -> **능동 부저(active)** 용
 *   구간 B (LED 꺼짐) : tone() / noTone()       -> **수동 부저(passive)** 용
 *
 * 어느 쪽에서 울리는지가 곧 답이다. 그 결과를 메인 스케치의
 *
 *     const bool ACTIVE_BUZZER = ...;
 *
 * 에 반영하면 된다.  A에서 울림 -> true,  B에서 울림 -> false.
 *
 * 둘 다 무음이면 소프트웨어가 아니라 하드웨어를 보라:
 *   - 부저 (+) 단자가 8번 핀에, (-) 단자가 GND에 들어갔는가
 *   - 3핀 모듈이면 VCC/GND/SIG 세 개가 모두 연결됐는가
 *   - 브레드보드 레일이 실제로 이어져 있는가 (가운데가 끊긴 보드가 많다)
 *   - 일부 3핀 모듈은 active-low다 — LOW일 때 운다. 그러면 구간 A에서
 *     '소리 남/안 남'이 뒤집혀 들린다(쉬는 구간에 울림).
 *
 * ⚠️ 부품이 부저가 아니라 **스피커**라면 핀에 직결하지 말 것.
 *
 *   멀티미터로 두 단자 저항을 재면 갈린다:
 *       4~8Ω    -> 스피커        (tone() 필요 + 직렬 저항 필수)
 *       16~42Ω  -> 수동 부저      (tone() 필요)
 *       수백 Ω~ -> 능동 부저      (digitalWrite로 충분)
 *
 *   8Ω을 5V에 직결하면 600mA 이상이 흐른다. 아두이노 핀 정격은 20mA(최대 40mA)이므로
 *   소리가 안 나거나 약할 뿐 아니라 핀/MCU가 손상될 수 있다.
 *   최소한 100~220Ω을 직렬로 넣고, 음량이 부족하면 트랜지스터로 구동할 것.
 *   스피커는 DC로는 울지 않으므로 구간 B(tone)에서만 소리가 난다.
 */

const int BUZZER_PIN = 8;     // 메인 스케치와 같은 핀
const int TONE_HZ    = 2000;
const int LED_PIN    = 13;    // 보드 내장 LED — 지금 어느 구간인지 표시

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  // ---- 구간 A: 능동 부저 방식 (LED 켬) ----
  digitalWrite(LED_PIN, HIGH);
  for (int i = 0; i < 5; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(300);
    digitalWrite(BUZZER_PIN, LOW);
    delay(300);
  }

  delay(1000);   // 구간 사이 정적 — 어디까지가 A인지 귀로 구분하기 위함

  // ---- 구간 B: 수동 부저 방식 (LED 끔) ----
  digitalWrite(LED_PIN, LOW);
  for (int i = 0; i < 5; i++) {
    tone(BUZZER_PIN, TONE_HZ);
    delay(300);
    noTone(BUZZER_PIN);
    delay(300);
  }

  delay(1000);
}
