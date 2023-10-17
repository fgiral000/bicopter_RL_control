## Module to do inference with the trained agent and plot the results 
import serial
import threading
import queue
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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def setup_arduino():
    """Setup arduino and serial port"""
    arduino = serial.Serial('/dev/ttyACM0', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
    print("Correctamente conectado")

    time.sleep(3)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    return arduino

def animate(i, x, y1, y2, line1, line2):
    line1.set_data(x[:i], y1[:i])
    line2.set_data(x[:i], y2[:i])
    return line1, line2

def create_animation(x, y1, y2):
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], 'b-', label='Current Angle')
    line2, = ax.plot([], [], 'r-', label='Reference Angle')

    ax.set_xlim(min(x), max(x))
    ax.set_ylim(-60, 60)  # Adjust if your data range is different
    ax.legend()

    ani = animation.FuncAnimation(
        fig, animate, len(x), fargs=[x, y1, y2, line1, line2],
        interval=25, blit=True
    )

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # Save the animation as an MP4 file
    ani.save('my_animation.mp4', writer=writer)

    plt.show()



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
  
    data_storage = []
    current_step = 0
    current_step_counter = 0
    # unnorm_observation[0][18] = rt
    # print("Observation after reset:")
    # print(unnorm_observation)
    # observation = env.normalize_obs(unnorm_observation)
    logging.info("The current reference angle is:",)
    # print(rt)
    
    while True:

        try:
            current_step +=1
            current_step_counter+=1
            action, _ = model.predict(observation, deterministic=True)
            observation, rewards, done, info = env.step(action)
            # denorm_observation
            unnorm_observation = env.get_original_obs()
           
         
            #################################################### plotting ##########################################################

            current_angle = unnorm_observation[0][16]
            reference_angle = unnorm_observation[0][18]
            data_storage.append((current_step, current_angle, reference_angle))
            #############################################################################################################################





            if current_step_counter >= 100:
                new_theta_value = np.random.choice([-25.0,0.0,25.0])
                env.env_method("set_theta_reference", new_theta_value)         #possibles angles of reference to give to the agent
                print("The new theta reference is:", new_theta_value)
                current_step_counter = 0

        except KeyboardInterrupt:
            print("The process has been finalized by keyboard command")
            #Arduino setup and serial port are closed
            arduino_port.close()
            # Post-process the collected data to create an animation
            x, y1, y2 = zip(*data_storage)  # Unpack the data
            create_animation(x, y1, y2)
            break


