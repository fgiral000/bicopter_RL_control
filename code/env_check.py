from code.gym_env_balancin_v2 import ControlEnv
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import time
import serial
import logging

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit
from wrappers_from_rlzoo import ActionSmoothingWrapper, HistoryWrapper


def setup_arduino():
    """Funcion para hacer el septup del arduino"""
    arduino = serial.Serial('COM5', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
    print("Correctamente conectado")

    time.sleep(3)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    return arduino








if __name__ == "__main__":

    #############################################################################################################################
    #############################################################################################################################
    #############################################################################################################################
    ############################# ENVIRONMENT #######################################################################################

    #Se inicializa el entorno del arduino

    arduino_port = setup_arduino()
    logging.info("Enciende la fuente de alimentacion")
    logging.info("Espera 20 segundos hasta que todo el sistema este activo")
    # time.sleep(10)

    logging.info("El sistema se ha activado correctamente")

    input("Presiona la tecla enter cuando todo este preparado",)





    #Se establece el entorno de entrenamiento
    env = ControlEnv(arduino_port)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=500)
    env = ActionSmoothingWrapper(env, smoothing_coef=0.6)
    env = HistoryWrapper(env=env)
    
    # # #VecNormalize wrappers
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env,
                       training=True,
                       norm_obs=True,
                       norm_reward=True,
                       clip_obs=10)

    check_env(env, warn=True)