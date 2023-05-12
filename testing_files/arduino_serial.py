import serial
import time
import numpy as np

arduino = serial.Serial('COM5',9600)
time.sleep(2)

a = True
while a!="Comienzo de lectura":
    a = arduino.readline()
    a = a[:-1]
    a = a.decode('utf-8')
    print(a)

a = arduino.readline()
a = a[:-1]
a = a.decode('utf-8')
a = a + '.txt'
f = open('C:/Users/dgtss/ComsArduino/Constantes_PID/' + a,'w')
print("Abierto archivo de texto:", a)

b = arduino.readline()
b = b[:-1]
b = b.decode('utf-8')
f.write(b)
f.write('\n')

lectura = "0,0,0,0,1"

while lectura[-1]=="1":
    lectura = arduino.readline()
    lectura = lectura[:-1]
    lectura = lectura.decode('utf-8')
    f.write(lectura)
    f.write('\n')
    print(lectura)

arduino.close()
f.close()