import serial
import numpy as np
import time

import serial.tools.list_ports

serialInst = serial.Serial()

portVar = "COM5"

serialInst.baudrate = 9600
serialInst.port = portVar
serialInst.open()



while len(state_vector) < 33:
    if serialInst.in_waiting:
        packet = serialInst.readline()
        state_vector = packet.decode('utf').rstrip('\n')
        print(state_vector)
            

serialInst.close()

