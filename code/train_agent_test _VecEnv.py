import serial
import time
import logging
# import yaml
from gym.wrappers import FrameStack, NormalizeReward, NormalizeObservation, TimeLimit
from wrappers_from_rlzoo import ActionSmoothingWrapper, HistoryWrapper
import gym
import stable_baselines3
#import tensorflow as tf
import tensorboard
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
    time.sleep(10)

    logging.info("El sistema se ha activado correctamente")

    input("Presiona la tecla enter cuando todo este preparado",)





    #Se establece el entorno de entrenamiento
    env = ControlEnv(arduino_port)
    env = Monitor(env)
    # env = TimeLimit(env, max_episode_steps=500)
    env = ActionSmoothingWrapper(env, smoothing_coef=0.6)
    env = HistoryWrapper(env=env)
    
    #VecNormalize wrappers
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env,
                       training=True,
                       norm_obs=True,
                       norm_reward=True,
                       clip_obs=10)
    
    




    logging.info("Estableciendo entorno de entrenamiento")
    time.sleep(2)
    # env.reset()
    # time.sleep(2)





    input("Vuelve a presionar enter para que el agente se ejecute",)



    #############################################################################################################################
    #############################################################################################################################
    #############################################################################################################################
    ############################# AGENTE #######################################################################################
    # # Se definen las variables a monitorizar en Tensorboard
    r_callback = TensorboardCallback()
    parallel_callback = ParallelTrainCallback(gradient_steps=200)

    # #Se empieza con el entrenamiento del agente
    


    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64], qf = [64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256], qf=[256,256]),
        "small_med": dict(pi=[128, 128], vf=[256, 256], qf = [256, 256])
    }["small"]

    # policy_kw = dict(activation_fn = torch.nn.Tanh, net_arch = net_arch, n_quantiles = 20, log_std_init = -1)
    policy_kw = dict(activation_fn = torch.nn.Tanh, net_arch = net_arch)
    
    sac = TQC('MlpPolicy',
                env=env,
                learning_rate=3e-4,
                buffer_size=10000,
                batch_size=256,
                ent_coef='auto',
                gamma=0.99,
                tau=0.02,
                train_freq=128,
                gradient_steps=128,
                learning_starts=500,
                use_sde_at_warmup=False,
                use_sde=True,
                sde_sample_freq=64,
                policy_kwargs=dict(log_std_init=-3, net_arch=[64,64], n_critics = 2),
                tensorboard_log="tqc_testing_1",
                verbose = 2,
                seed = 68,
                )
    
    # sac = SAC.load("sac_model_trained_from_pretrained_50k.zip", env=env)

    TIME_STEPS = 40_000
    CALLBACKS = [r_callback, parallel_callback]
    
    sac.learn(total_timesteps = TIME_STEPS, callback = CALLBACKS, tb_log_name="tqc_state_space_VecEnv")

    sac.save(path="tqc_model_test_VecEnv")
    sac.save_replay_buffer("replay_buffer_tqc_training__VecEnv")
    env.save("vec_normalize.pkl")

    #Se finaliza todo el setup del arduino
    # env.reset()
    arduino_port.close()



