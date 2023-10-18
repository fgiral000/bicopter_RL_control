import serial
import time
import logging
# import yaml
from gymnasium.wrappers import TimeLimit
from wrappers_from_rlzoo import ActionSmoothingWrapper, HistoryWrapper
import gymnasium as gym
import stable_baselines3

from gym_env_balancin_v2 import ControlEnv
from gym_env_balancin_v2 import TensorboardCallback
from callbacks_from_rlzoo import ParallelTrainCallback
from stable_baselines3 import SAC
from sb3_contrib import TQC
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


######### SOFT ACTOR-CRITIC TRAINING TEST CODE ###########

def setup_arduino():
    """Function to set up the arduino"""
    arduino = serial.Serial('/dev/ttyACM0', 9600) # Replace 'COM3' with the serial port of your Arduino
    print("Successfully connected")

    time.sleep(3)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    return arduino








if __name__ == "__main__":

    #############################################################################################################################
    #############################################################################################################################
    #############################################################################################################################
    ############################# ENVIRONMENT #######################################################################################

    #Names of the starting files
    MODEL_NAME_LOAD_FROM = "../tqc_model_3targets"
    MODEL_BUFFER_LOAD_FROM = "../replay_buffer_tqc_training_3targets.pkl"
    VEC_ENV_LOAD_FROM = "../vec_normalize_3targets.pkl"

    MODEL_NAME_NEW = "../tqc_model_3targets_nostop"
    MODEL_BUFFER_NEW = "../replay_buffer_tqc_training_3targets_nostop.pkl"
    VEC_ENV_NEW = "../vec_normalize_3targets_nostop.pkl"

    # MODEL_NAME_NEW = "../tqc_model_1targets"
    # MODEL_BUFFER_NEW = "../replay_buffer_tqc_training_1targets.pkl"
    # VEC_ENV_NEW = "../vec_normalize_1targets.pkl"


    #Arduino environment is initialized

    arduino_port = setup_arduino()
    logging.info("Turn on the power supply")
    logging.info("Wait 20 seconds until the entire system is active")
    time.sleep(10)

    logging.info("The system has been activated correctly")

    input("Press the enter key when everything is ready",)





    #The training environment is established
    env = ControlEnv(arduino_port)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=500)
    env = ActionSmoothingWrapper(env, smoothing_coef=0.6)
    env = HistoryWrapper(env=env)
    
    # # #VecNormalize wrappers
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env,
    #                    training=True,
    #                    norm_obs=True,
    #                    norm_reward=True,
    #                    clip_obs=10)
    
    
    env = VecNormalize.load(VEC_ENV_LOAD_FROM, 
                            venv=env)
    env.training = True



    logging.info("Setting up training environment")
    time.sleep(2)
    # env.reset()
    # time.sleep(2)





    input("Press enter again to execute the agent",)



    #############################################################################################################################
    #############################################################################################################################
    #############################################################################################################################
    ############################# AGENT #######################################################################################
    # # Variables to be monitored in Tensorboard are defined
    r_callback = TensorboardCallback()
    parallel_callback = ParallelTrainCallback(gradient_steps=200)

    # #Training of the agent begins
    


    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64], qf = [64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256], qf=[256,256]),
        "small_med": dict(pi=[128, 128], vf=[256, 256], qf = [256, 256])
    }["small"]

    # policy_kw = dict(activation_fn = torch.nn.Tanh, net_arch = net_arch, n_quantiles = 20, log_std_init = -1)
    policy_kw = dict(activation_fn = torch.nn.Tanh, net_arch = net_arch)
    
    # sac = TQC('MlpPolicy',
    #             env=env,
    #             learning_rate=3e-4,
    #             buffer_size=10000,
    #             batch_size=256,
    #             ent_coef='auto',
    #             gamma=0.99,
    #             tau=0.02,
    #             train_freq=128,
    #             gradient_steps=128,
    #             learning_starts=500,
    #             use_sde_at_warmup=False,
    #             use_sde=True,
    #             sde_sample_freq=64,
    #             policy_kwargs=dict(log_std_init=-3, net_arch=[64,64], n_critics = 2),
    #             tensorboard_log="../TQC_algo",
    #             verbose = 2,
    #             seed = 68,
    #             )
    
    # sac = SAC.load("sac_model_trained_from_pretrained_50k.zip", env=env, )
    #####Un-comment when you want to train from a pre-trained model
    sac = TQC.load(MODEL_NAME_LOAD_FROM, custom_objects={"verbose":2, "learning_starts":0, "tensorboard_log":"../TQC_algo"}, env=env)
    sac.load_replay_buffer(MODEL_BUFFER_LOAD_FROM)
    sac.set_env(env=env)

    TIME_STEPS = 30_000
    CALLBACKS = [r_callback, parallel_callback]
    
    sac.learn(total_timesteps = TIME_STEPS, callback = CALLBACKS, tb_log_name="tqc_3targets")

    sac.save(path=MODEL_NAME_NEW)
    sac.save_replay_buffer(MODEL_BUFFER_NEW)
    env.save(VEC_ENV_NEW)

    #Arduino setup is finalized
    # env.reset()
    arduino_port.close()





