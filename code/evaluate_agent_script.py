import serial
import time
import logging
# import yaml
from gymnasium.wrappers import TimeLimit
from wrappers_from_rlzoo import ActionSmoothingWrapper, HistoryWrapper
import gymnasium as gym
import stable_baselines3
# import yaml
from gym_env_balancin_v2 import ControlEnv
from gym_env_balancin_v2 import TensorboardCallback
from callbacks_from_rlzoo import ParallelTrainCallback
from stable_baselines3 import SAC
from sb3_contrib import TQC
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


######### CODIGO PRUEBA DE ENTRENAMIENTO DE SOFT ACTOR-CRITIC ###########

def setup_arduino():
    """Funcion para hacer el septup del arduino"""
    arduino = serial.Serial('/dev/ttyACM0', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
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
    time.sleep(10)

    logging.info("El sistema se ha activado correctamente")

    input("Presiona la tecla enter cuando todo este preparado",)


    MODEL_NAME_NEW = "../tqc_model_3targets"
    MODEL_BUFFER_NEW = "../replay_buffer_tqc_training_3targets.pkl"
    VEC_ENV_NEW = "../vec_normalize_3targets.pkl"


    #Se establece el entorno de entrenamiento
    env = ControlEnv(arduino_port)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=200)
    env = ActionSmoothingWrapper(env, smoothing_coef=0.6)
    env = HistoryWrapper(env=env)
    
    #VecNormalize wrappers
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(load_path=VEC_ENV_NEW, venv=env)
    env.training = False
    env.norm_reward = False
    
    

    logging.info("Estableciendo entorno de entrenamiento")
    time.sleep(2)
    # env.reset()
    # time.sleep(2)





    input("Vuelve a presionar enter para que el agente se ejecute",)



    #############################################################################################################################
    #############################################################################################################################
    #############################################################################################################################
    ############################# AGENTE #######################################################################################
    
    model = TQC.load(MODEL_NAME_NEW)
    model.set_env(env)
    model.set_parameters(MODEL_NAME_NEW)
    mean_reward, std_reward = evaluate_policy(model, env, deterministic=True, n_eval_episodes=3)

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    #Se finaliza todo el setup del arduino
    # env.reset()
    arduino_port.close()
