## Module to do inference with the trained agent and plot the results 
import serial
import time
import logging
from gymnasium.wrappers import TimeLimit
from code.wrappers_from_rlzoo import ActionSmoothingWrapper, HistoryWrapper
import gymnasium as gym
import stable_baselines3
from code.gym_env_balancin_v2 import ControlEnv
from code.callbacks_from_rlzoo import ParallelTrainCallback
from stable_baselines3 import SAC
from sb3_contrib import TQC
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def setup_arduino():
    """Funcion para hacer el septup del arduino"""
    arduino = serial.Serial('COM5', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
    print("Correctamente conectado")

    time.sleep(3)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    return arduino


if __name__ == "__main__":

    #Load the environment

    #############################################################################################################################
    #############################################################################################################################
    #############################################################################################################################
    ############################# ENVIRONMENT #######################################################################################

    #Se inicializa el entorno del arduino
    arduino_port = setup_arduino()
    logging.info("Enciende la fuente de alimentacion")
    logging.info("Espera 10 segundos hasta que todo el sistema este activo")
    time.sleep(10)
    logging.info("El sistema se ha activado correctamente")
    input("Presiona la tecla enter cuando todo este preparado",)

    #Se establece el entorno de entrenamiento
    env = ControlEnv(arduino_port)
    env = Monitor(env)
    # env = TimeLimit(env, max_episode_steps=200)
    env = ActionSmoothingWrapper(env, smoothing_coef=0.6)
    env = HistoryWrapper(env=env)
    #VecNormalize wrappers
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(load_path="./vec_normalize_3targets.pkl", venv=env)
    env.training = False
    env.norm_reward = False
    logging.info("Estableciendo entorno de entrenamiento")
    input("Vuelve a presionar enter para que el agente se ejecute",)





    #Load the pre-trained agent
    model = TQC.load("./tqc_model_3targets")
    model.set_env(env)
    model.set_parameters("./tqc_model_3targets")



    #Using stable-baselines3 to use the pre-trained policy in inference
    
    reference_theta = [-10.0,0.0,10.0]          #possibles angles of reference to give to the agent

    for rt in reference_theta:
        observation = env.reset()
        # unnorm_observation = env.unnormalize_obs(observation)
        # # print(unnorm_observation)
        # unnorm_observation[0][-4] = 0.0
        # observation = env.normalize_obs(unnorm_observation)
        logging.info("The current reference angle is:",)
        # print(rt)
        
        for _ in range(500):

            try:
                action, _ = model.predict(observation, deterministic=True)
                observation, rewards, done, info = env.step(action)
                # unnorm_observation = env.unnormalize_obs(observation)
                # print(unnorm_observation[0][-4])
                # print(f"Current angle: {observation[0][-6]}, Error track: {abs(observation[0][-6]-observation[0][-4])}")

            except KeyboardInterrupt:
                print("The process has been finalized by keyboard command")
                #Arduino setup and serial port are closed
                arduino_port.close()




