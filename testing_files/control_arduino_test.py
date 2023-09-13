import serial
import numpy as np
import time
import random

# Open arduinoial port
# arduino = serial.Serial('COM5', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
# print("Correctamente conectado")

def send_action():
    pass



def get_state():
    pass


def sendSequenceToArduino(action_array):
    """funcion para enviar acciones de los motores al arduino"""
    actions = np.round(action_array, decimals=1)
    # Convert sequence to string
    sequenceString = '$' + str(actions[0]) + str(actions[1]) + '#' # Assumes sequence is a 2D array with shape (1, 2)
    #print(sequenceString)
    # Send sequence to Arduino
    arduino.write(sequenceString.encode())
    

def read_serial_data():
    arduino.reset_input_buffer()
    #time.sleep(1)
    #data = ''
    while True:
        if arduino.in_waiting > 0:
            #time.sleep(1)
            data = arduino.readline()
            data = data[:-1]
            data = data.decode('utf-8')
            
            print(data)

            if data[0] == '&' and data[-1] == '@':
                data = data.strip('&@')
                values = data.split(',')

                # arduino.write('ACK'.encode())


                break
            #data += arduino.read().decode()
            #data = arduino.readline().decode('utf').rstrip('@')
            #print(data)
            # if data[-1] == '@':
            #     break
    #arduino.close()c
    return [float(value) for value in values]
    #return data

def parse_serial_data(data):
    data = data.strip('&@')
    values = data.split(',')
    return [float(value) for value in values]


def get_data():
    # Wait until the start character '<' is received
    while True:
        if arduino.read().decode() == '<':
            break

    # Read the data until the end character '>' is received
    data = ''
    while True:
        char = arduino.read().decode()
        if char == '>':
            arduino.write('?'.encode())
            break
        data += char

    # Convert the data string to a list of integers
    data_list = list(map(float, data.split(',')))
    return data_list

# arduino = serial.Serial('COM5', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
# print("Correctamente conectado")
# time.sleep(5)
# while True:
#     try:
#         data = read_serial_data()
#         print(data)
#         # values = parse_serial_data(data)
#         # print(values)
#     except KeyboardInterrupt:
#         arduino.close()
#         print("Puerto serie desconectado")


if __name__ == "__main__":
    arduino = serial.Serial('COM5', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
    print("Correctamente conectado")

    input("Presiona la tecla enter cuando este preparado",)
    while True:
        #time.sleep(2)
        print("Ya estas aqui")
        print(arduino.readline().decode())
        try:
            for array in [[1100.0,1100.0], [1300.0, 1150.0], [1100.0,1300.0]]:
                sendSequenceToArduino([random.uniform(1000.0, 1300.0), random.uniform(1000.0, 1300.0)])
                print("has enviado los datos:", array)
                # data = read_serial_data()
                # time.sleep(1)
                data = get_data()
                print(data)
                #time.sleep(2)

        except KeyboardInterrupt:
            sendSequenceToArduino([1000.0,1000.0])
            arduino.close()

