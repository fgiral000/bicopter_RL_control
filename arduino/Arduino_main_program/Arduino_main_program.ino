#include <Wire.h>
#include <Servo.h>
#include "I2Cdev.h"
#include "MPU6050.h"
#include <cvzone.h>

Servo right_prop;
Servo left_prop;
SerialData serialData;

// La dirección del MPU6050 puede ser 0x68 o 0x69, dependiendo 
// del estado de AD0. Si no se especifica, 0x68 estará implicito
MPU6050 sensor(0x69);


/* Valores RAW (sin procesar) del acelerometro y giroscopio en los ejes x,y,z.
 La MPU6050 envia datos de 16 bits asi que creamos las variables de las lecturas*/
int ax, ay, az;
int gx, gy, gz;


float rad_to_deg = 180/3.141592654;      // Cambio de unidades de radianes a grados
long t1,t2;                              // Tiempos auxiliares para forzar que el bucle se ejecute a 20Hz
float dt;                                // Diferencia de tiempo entre ejecuciones sucesivas, en segundos
float ang_x;                             // Angulo del balancin
float ang_x_prev;                        // Angulo de la lectura anterior
float accel_ang_x;                       // Aceleracion angular 

// Variables del PID y de los motores
float PID, pwmLeft, pwmRight, error, previous_error;
float pid_p=0;
float pid_i=0;
float pid_d=0;

float leftMotor=1000.0;
float rightMotor=1000.0;

// Tiempos de ejecucion
long t_aux;

//////////////CONSTANTES DEL PID///////////////
double kp=1;//3.55
double ki=0.000;//0.003
double kd=0.0;//2.05
///////////////////////////////////////////////

/* Los motores funcionan con entradas PWM, con
una entrada de 1000 us dan un 0% de la potencia
y con 2000 us el 100% */

#define MAX_SIGNAL 2000
#define MIN_SIGNAL 1000

double throttle=1200;                   // Valor iniciar de entrada de los motores

float desired_angle = 0;                // Angulo donde se quiere estabilizar

void setup() {
  Serial.begin(9600);                   // Inicio puerto serial
  Wire.begin();                         // Inicio I2C  
  sensor.initialize();                  // Inicio el sensor

  // Offset de la IMU
  sensor.setXAccelOffset(823);
  sensor.setYAccelOffset(-1352);
  sensor.setZAccelOffset(1861);
  sensor.setXGyroOffset(55);
  sensor.setYGyroOffset(87);
  sensor.setZGyroOffset(-9);

  right_prop.attach(5);                 // Establecer el motor derecho en el pin 3
  left_prop.attach(3);                  // Establecer el motor izquierdo en el pin 5

  if (sensor.testConnection()) Serial.println("\nSensor iniciado correctamente");
  else Serial.println("\nError al iniciar el sensor");

  
  // Setup de los motores
  Serial.println("Enciende la fuente de alimentacion");
  delay(5000);

  // Se envia la señal maxima
  Serial.print("Enviando senal maxima: (");Serial.print(MAX_SIGNAL);Serial.print(" us)");Serial.print("\n");
  right_prop.writeMicroseconds(MAX_SIGNAL);
  left_prop.writeMicroseconds(MAX_SIGNAL);
  
  delay(1000);

  // Se envia la señal minima
  Serial.print("Enviando senal minima: (");Serial.print(MIN_SIGNAL);Serial.print(" us)");Serial.print("\n");
  right_prop.writeMicroseconds(MIN_SIGNAL);
  left_prop.writeMicroseconds(MIN_SIGNAL);

  Serial.println("Calibrado de los motores realizado");
  Serial.println("----");

  delay(5000);
  Serial.print("Comienzo de lectura");Serial.print("\n");
  Serial.print("PID_3");Serial.print("\n");
  Serial.print(kp);Serial.print(",");Serial.print(ki,3);Serial.print(",");Serial.print(kd);Serial.print(",");
  Serial.print(0);Serial.print(",");Serial.print(0);Serial.print(",");Serial.print(0);
  Serial.print("\n");
  t2=50+millis();
  t_aux=millis();


  // Leer las aceleraciones y velocidades angulares
  sensor.getAcceleration(&ax, &ay, &az);
  sensor.getRotation(&gx, &gy, &gz);
  
  // Queremos conseguir una frecuencia de 20Hz (50ms)
  // t1 = millis();
  // delay(50-(t1-t2));
  // t2 = millis();

  dt = 50.0/1000.0;
  
  //Calcular los angulos con acelerometro
  accel_ang_x=atan(ay/sqrt(pow(ax,2) + pow(az,2)))*rad_to_deg;
  
  //Calcular angulo de rotacion con giroscopio y filtro complemento  
  ang_x = 0.8*(ang_x_prev+(gx/131)*dt) + 0.2*accel_ang_x;
  ang_x_prev=ang_x;




  // delay(20000);
}//end of setup void



void loop() {
/////////////////////////////I M U/////////////////////////////////////  
  received_motor_data();
  left_prop.writeMicroseconds(int(leftMotor));
  right_prop.writeMicroseconds(int(rightMotor));



  // Leer las aceleraciones y velocidades angulares
  sensor.getAcceleration(&ax, &ay, &az);
  sensor.getRotation(&gx, &gy, &gz);
  
  // Queremos conseguir una frecuencia de 20Hz (50ms)
  // t1 = millis();
  // delay(50-(t1-t2));
  // t2 = millis();

  dt = 50.0/1000.0;
  
  //Calcular los angulos con acelerometro
  accel_ang_x=atan(ay/sqrt(pow(ax,2) + pow(az,2)))*rad_to_deg;
  
  //Calcular angulo de rotacion con giroscopio y filtro complemento  
  ang_x = 0.8*(ang_x_prev+(gx/131)*dt) + 0.2*accel_ang_x;
  ang_x_prev=ang_x;


// Serial.flush();
//  delay(5000);  

  // // Send data to Python
  // Serial.print("<");
  // Serial.print(ang_x);
  // Serial.print(",");
  // Serial.print(gx);
  // Serial.print(",");
  // Serial.print(leftMotor);
  // Serial.print(",");
  // Serial.print(rightMotor);
  // Serial.println(">");

  // // Wait for Python to acknowledge the received data
  // while (!Serial.available()) {
  //   delay(10);
  //   }
  // // Clear the buffer
  // while (Serial.available()) {
  //   char c = Serial.read();
  //   if (c == '?') {
  //     break;
  //   }
  // }
// delay(3000);
receive_confirmation();

}

void received_motor_data() {
  String sequence = "";
  while (true) {
    if (Serial.available() > 0) {
      char c = Serial.read();
      if (c == '$') {
        sequence = "";
      } else if (c == '#') {
        if (sequence.length() == 12) {
          leftMotor = sequence.substring(0, 6).toFloat();
          rightMotor = sequence.substring(6, 12).toFloat();
          break;
        } else {
          sequence = "";
        }
      } else if (sequence.length() < 12) {
        sequence += c;
      }
    }
  }

}

void receive_confirmation() {

  bool confirmed = false;
  while (!confirmed) {
    // Send data to Python
    Serial.print("<");
    Serial.print(ang_x);
    Serial.print(",");
    Serial.print(gx/131);
    Serial.print(",");
    Serial.print(leftMotor);
    Serial.print(",");
    Serial.print(rightMotor);
    Serial.println(">");

    // Wait for Python to acknowledge the received data
    while (!Serial.available()) {
      delay(10);
      }
    // Clear the buffer
    while (Serial.available()) {
      char c = Serial.read();
      if (c == '?') {
        confirmed = true;
        break;
      }
    }
  }

}

// void receive_confirmation() {

//   bool confirmed = false;
//   while (!confirmed) {
//     // Send data to Python
//     Serial.print("<");
//     Serial.print(ang_x);
//     Serial.print(",");
//     Serial.print(gx);
//     Serial.print(",");
//     Serial.print(leftMotor);
//     Serial.print(",");
//     Serial.print(rightMotor);
//     Serial.println(">");

//     // Clear the buffer
//     if (Serial.available()) {
//       char c = Serial.read();
//       if (c == '?') {
//         confirmed = true;
//         break;
//       }
//     }
//   }

// }