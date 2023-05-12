import serial

ser = serial.Serial('COM5', 9600) # Change the serial port to the appropriate port name on your computer

def get_data():
    # Wait until the start character '<' is received
    while True:
        if ser.read().decode() == '<':
            break

    # Read the data until the end character '>' is received
    data = ''
    while True:
        char = ser.read().decode()
        if char == '>':
            ser.write('?'.encode())
            break
        data += char

    # Convert the data string to a list of integers
    data_list = list(map(float, data.split(',')))
    return data_list


# def get_data():
#     # Wait until the start character '<' is received
#     while True:
#         if arduino.in_waiting > 0:
#             print("Ha entrado en la funcion de leer datos")
#             while True:
#                 print("esperando a que llegue el primer caracter")
#                 print(arduino.read().decode())
#                 if arduino.read().decode() == '<':
#                     print("el primer caracter ha llegado")
#                     break
#                 print("No ha llegado el primer caracter")

#             # Read the data until the end character '>' is received
#             data = ''
#             while True:
#                 char = arduino.read().decode()
#                 if char == '>':
#                     print("se ha leido el ultimo caracter")
#                     arduino.write('?'.encode())
#                     break
#                 data += char

#             # Convert the data string to a list of integers
#             data_list = list(map(float, data.split(',')))
#             return data_list


# Request data from Arduino
while True:
    data = get_data()
    # Do something with the data
    print(data)
