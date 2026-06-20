#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define BT Serial1

// Motor pins
#define IN1 4
#define IN2 5
#define IN3 6
#define IN4 7

// LED
#define RED_LED 40

// Ultrasonic
#define TRIG_PIN 30
#define ECHO_PIN 31
#define OBSTACLE_DISTANCE 30

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN 150
#define SERVOMAX 600

// ===== Servo Channels =====
#define SERVO_1 15    // A
#define SERVO_2 14    // C
#define SERVO_3 13    // C (mirror)
#define SERVO_4 12    // D
#define SERVO_5 11    // E
#define SERVO_6 10    // G
#define SERVO_7 9     // H

// Servo motion arrays
int currentAngle[16];
int targetAngle[16];

// Motion state
char currentMotion = 'S';

// ================= SETUP =================
void setup() {
  Serial.begin(9600);
  BT.begin(9600);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(RED_LED, OUTPUT);
  digitalWrite(RED_LED, LOW);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  pwm.begin();
  pwm.setPWMFreq(50);

  // Initialize servos
  for (int i = 0; i < 16; i++) {
    currentAngle[i] = 90;
    targetAngle[i] = 90;
  }

  Serial.println("System Ready");
}

// ================= MOTOR =================
void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  digitalWrite(RED_LED, LOW);
}

void forward() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  digitalWrite(RED_LED, LOW);
}

void backward() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  digitalWrite(RED_LED, HIGH);
}

void left() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  digitalWrite(RED_LED, LOW);
}

void right() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  digitalWrite(RED_LED, LOW);
}

// ================= ULTRASONIC =================
long getDistanceRaw() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return 0;

  return duration * 0.034 / 2;
}

long getStableDistance() {
  long d1 = getDistanceRaw();
  delay(10);
  long d2 = getDistanceRaw();
  delay(10);
  long d3 = getDistanceRaw();

  if (d1 == 0) d1 = d2;
  if (d2 == 0) d2 = d3;
  if (d3 == 0) d3 = d2;

  return (d1 + d2 + d3) / 3;
}

// ================= SERVO =================
int angleToPWM(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

// ================= PARSER =================
String input = "";

void loop() {

  long distance = getStableDistance();
  bool obstacleAhead = (distance > 0 && distance < OBSTACLE_DISTANCE);

  Serial.print("Distance: ");
  Serial.println(distance);

  // ===== CONTINUOUS SAFETY =====
  if (currentMotion == 'F' && obstacleAhead) {
    Serial.println("Obstacle -> Auto Back");

    backward();
    delay(300);
    stopMotors();
    currentMotion = 'S';
  }

  // ===== BLUETOOTH INPUT =====
  while (BT.available()) {
    char c = BT.read();

    if (c == 'F') {
      if (!obstacleAhead) {
        forward();
        currentMotion = 'F';
      } else {
        stopMotors();
        currentMotion = 'S';
      }
    }

    else if (c == 'B') {
      backward();
      currentMotion = 'B';
    }

    else if (c == 'L') {
      left();
      currentMotion = 'L';
    }

    else if (c == 'R') {
      right();
      currentMotion = 'R';
    }

    else if (c == 'S') {
      stopMotors();
      currentMotion = 'S';
    }

    else if (c >= 'a' && c <= 'z') {
      stopMotors();
      currentMotion = 'S';
    }

    // ===== SLIDERS =====
    else {
      input += c;

      if (input.length() >= 3) {
        char startChar = input.charAt(0);
        char endChar = input.charAt(input.length() - 1);

        if (startChar == endChar) {

          int value = input.substring(1, input.length() - 1).toInt();

          switch (startChar) {
            case 'A': targetAngle[SERVO_1] = value; break;

            case 'C':
              targetAngle[SERVO_2] = value;
              targetAngle[SERVO_3] = 180 - value;
              break;

            case 'D': targetAngle[SERVO_4] = value; break;
            case 'E': targetAngle[SERVO_5] = value; break;
            case 'G': targetAngle[SERVO_6] = value; break;
            case 'H': targetAngle[SERVO_7] = value; break;
          }

          input = "";
        }
      }
    }
  }

  // ===== SMOOTH SERVO UPDATE =====
  for (int ch = 9; ch <= 15; ch++) {

    if (currentAngle[ch] < targetAngle[ch]) {
      currentAngle[ch] += 1;
    }
    else if (currentAngle[ch] > targetAngle[ch]) {
      currentAngle[ch] -= 1;
    }

    pwm.setPWM(ch, 0, angleToPWM(currentAngle[ch]));
  }

  delay(10); // smoothness control
}
