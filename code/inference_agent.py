## Module to do inference with the trained agent and plot the results 
import serial
import time
import logging
from gymnasium.wrappers import TimeLimit
from wrappers_from_rlzoo import ActionSmoothingWrapper, HistoryWrapper
import gymnasium as gym
import stable_baselines3
from gym_env_balancin_v2 import ControlEnv
from callbacks_from_rlzoo import ParallelTrainCallback
from stable_baselines3 import SAC
from sb3_contrib import TQC
import torch
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def setup_arduino():
    """Funcion para hacer el septup del arduino"""
    arduino = serial.Serial('/dev/ttyACM0', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
    print("Correctamente conectado")

    time.sleep(3)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    return arduino


if __name__ == "__main__":

    #Load the environment
    MODEL_NAME_NEW = "../models/model/tqc_model_3targets_nostop"
    MODEL_BUFFER_NEW = "../models/replay_buffer/replay_buffer_tqc_training_3targets_nostop.pkl"
    VEC_ENV_NEW = "../models/vec_envs/vec_normalize_3targets_nostop.pkl"

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
    env = VecNormalize.load(load_path=VEC_ENV_NEW, venv=env)
    env.training = False
    env.norm_reward = False
    logging.info("Estableciendo entorno de entrenamiento")
    input("Vuelve a presionar enter para que el agente se ejecute",)





    #Load the pre-trained agent
    model = TQC.load(MODEL_NAME_NEW)
    model.set_env(env)
    model.set_parameters(MODEL_NAME_NEW)


    # env.get_original_obs()
    

    #Using stable-baselines3 to use the pre-trained policy in inference
    
    
    observation = env.reset()
    # unnorm_observation = env.get_original_obs()
    # # print(unnorm_observation)
    # # unnorm_observation = env.unnormalize_obs(observation)
    # # # print(unnorm_observation)
    current_step = 0
    
    # unnorm_observation[0][18] = rt
    # print("Observation after reset:")
    # print(unnorm_observation)
    # observation = env.normalize_obs(unnorm_observation)
    logging.info("The current reference angle is:",)
    # print(rt)
    
    while True:

        try:
            current_step +=1
            action, _ = model.predict(observation, deterministic=True)
            observation, rewards, done, info = env.step(action)
            # denorm_observation = env.unnormalize_obs(observation)
            # unnorm_observation = env.get_original_obs()
            # print("Normalized observation")
            # print(observation)
            # print("Original observation")
            # print(unnorm_observation)
            # # print("Denorm obervation")
            # # print(denorm_observation)
            # # print(unnorm_observation[0][-4])
            # unnorm_observation[0][18] = rt
            # observation = env.normalize_obs(unnorm_observation)
            # print(f"Current angle: {observation[0][-6]}, Error track: {abs(observation[0][-6]-observation[0][-4])}")
            if current_step >= 100:
                new_theta_value = np.random.choice([-25.0,0.0,25.0])
                env.env_method("set_theta_reference", new_theta_value)         #possibles angles of reference to give to the agent
                print("The new theta reference is:", new_theta_value)
                current_step = 0

        except KeyboardInterrupt:
            print("The process has been finalized by keyboard command")
            #Arduino setup and serial port are closed
            arduino_port.close()




