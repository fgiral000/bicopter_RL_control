import d3rlpy
import gym 
from gym_env_balancin_v2 import ControlEnv
import serial
import logging
import time
from gym.wrappers import NormalizeObservation, NormalizeReward

def setup_arduino():
    """Funcion para hacer el septup del arduino"""
    arduino = serial.Serial('COM5', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
    print("Correctamente conectado")

    time.sleep(3)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    return arduino



if __name__ == "__main__":

     #Se inicializa el entorno del arduino

    arduino_port = setup_arduino()
    logging.info("Enciende la fuente de alimentacion")
    logging.info("Espera 20 segundos hasta que todo el sistema este activo")
    time.sleep(10)

    logging.info("El sistema se ha activado correctamente")

    input("Presiona la tecla enter cuando todo este preparado",)
    #Se establece el entorno de entrenamiento
    env = ControlEnv(arduino_port)
    # env = FrameStack(env,num_stack=2)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)

    logging.info("Estableciendo entorno de entrenamiento")
    time.sleep(2)
    env.reset()
    time.sleep(2)

    input("Vuelve a presionar enter para que el agente se ejecute",)

    ##############################################################################################################
    ###############################################################################################################
    # setup algorithm
    random_policy = d3rlpy.algos.RandomPolicy()

    # prepare experience replay buffer
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

    # start data collection
    random_policy.collect(env, buffer, n_steps=100000)

    # export as MDPDataset
    dataset = buffer.to_mdp_dataset()

    # save MDPDataset
    dataset.dump("random_policy_dataset.h5")


    #Se finaliza todo el setup del arduino
    env.reset()
    arduino_port.close()
