import serial
import time
import logging
import yaml
from gym.wrappers import FrameStack, NormalizeReward, NormalizeObservation
import gym
import stable_baselines3
#import tensorflow as tf
import tensorboard
import yaml
from code.gym_env_balancin_v2 import ControlEnv
from code.gym_env_balancin_v2 import TensorboardCallback
from stable_baselines3 import SAC
from sb3_contrib import TQC
import torch
import d3rlpy
from d3rlpy.algos import AWAC, CQL
from d3rlpy.wrappers.sb3 import SB3Wrapper


######### CODIGO PRUEBA DE ENTRENAMIENTO DE SOFT ACTOR-CRITIC ###########

def setup_arduino():
    """Funcion para hacer el septup del arduino"""
    arduino = serial.Serial('COM5', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
    print("Correctamente conectado")

    time.sleep(3)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    return arduino



def random_agent(env, episodes = 2):
    """Agente que toma acciones aleatorias que sirve para testear el entorno"""

    agente = SAC('MlpPolicy', env)
    observation = env.reset()
    #Se inicializa una lista para almacenar los rewards de un episodio
    ep_returns = []

    for _ in range(episodes):
        observation = env.reset()
        done = False
        rewards = []
        while not done:
            action, _ = agente.predict(observation, deterministic=False) # Tomar una acción aleatoria
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            print(observation, action)
            # print(observation)
            # print(len(rewards))

def inference_agent(env, episodes):
    """Agente que toma acciones aleatorias que sirve para testear el entorno"""

    agente = SAC.load('C:/Users/B1500/OneDrive/Escritorio/repo_bicopter_RL/bicopter_RL_control/model_trained_v00.zip')
    observation = env.reset()
    #Se inicializa una lista para almacenar los rewards de un episodio
    ep_returns = []


    for _ in range(episodes):
        observation = env.reset()
        done = False
        rewards = []
        while not done:
            action, _ = agente.predict(observation, deterministic=True) # Tomar una acción aleatoria
            new_observation, reward, done, _ = env.step(action)
            observation = new_observation
            rewards.append(reward)
            print(reward)


def read_hyperparams():
    # Carga el archivo YAML
    with open('C:/Users/dgtss/Bicopter_RL/bicopter_RL_control/hyperparam.yaml', 'r') as f:
        hiperparametros = yaml.safe_load(f)

    # Accede a los hiperparámetros como un diccionario
    n_timesteps = hiperparametros['n_timesteps']
    policy = hiperparametros['policy']
    batch_size = hiperparametros['batch_size']
    learning_rate = hiperparametros['learning_rate']
    buffer_size = hiperparametros['buffer_size']
    ent_coef = hiperparametros['ent_coef']
    gamma = hiperparametros['gamma']
    tau = hiperparametros['tau']
    train_freq = hiperparametros['train_freq']
    gradient_steps = hiperparametros['gradient_steps']
    learning_starts = hiperparametros['learning_starts']
    policy_kwargs = hiperparametros['policy_kwargs']

    hyperparams = {
        #"n_timesteps": n_timesteps,
        "policy": policy,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "policy_kwargs": policy_kwargs,
    }


    return hyperparams

def read_hyperparmeters():
    with open('hyperparam.yaml') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)  # config is dict

    return config


def train_agent(env, time_steps,input_callback):
    """Funcion para entrenar el agente"""
    #hyperparams = read_hyperparams()

    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64], qf = [64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256], qf=[256,256]),
    }["small"]

    policy_kw = dict(activation_fn = torch.nn.Tanh, net_arch = net_arch)

    # sac = SAC('MlpPolicy',
    #             env=env,
    #             batch_size= 512,
    #             learning_rate= 3e-4,
    #             buffer_size= 10000,
    #             ent_coef= 'auto',
    #             gamma= 0.90,
    #             tau = 0.01,
    #             train_freq= 8,
    #             gradient_steps= 8,
    #             learning_starts= 2048,
    #             use_sde=True,
    #             sde_sample_freq = 512,
    #             verbose=2,
    #             seed=68,
    #             tensorboard_log="./sac_testing_v3",
    #             policy_kwargs = policy_kw,
    #             )
    
    # tqc = TQC('MlpPolicy',
    #             env=env,
    #             batch_size= 512,
    #             learning_rate= 3e-4,
    #             buffer_size= 10000,
    #             ent_coef= 'auto',
    #             gamma= 0.90,
    #             tau = 0.01,
    #             train_freq= 8,
    #             gradient_steps= 8,
    #             learning_starts= 2048,
    #             use_sde=True,
    #             sde_sample_freq = 512,
    #             verbose=2,
    #             seed=68,
    #             tensorboard_log="./sac_testing_v3",
    #             policy_kwargs = policy_kw,)

    # tqc = TQC.load("cql_v1.pt")
    cql = CQL()

    # initialize with environment
    cql.build_with_env(env)

    # load entire model parameters.
    cql.load_model("cql_v1.pt")

    # # Convertir el modelo de CQL en un modelo de SAC
    # tqc = SB3Wrapper(cql).wrap(TQC)

    # tqc.learn(total_timesteps = time_steps, callback = input_callback)

    # Entrenar online con el entorno
    cql.fit_online(env, n_steps=10000)

    cql.save_model("cql_model_finnetuned")




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

    # # Se definen las variables a monitorizar en Tensorboard
    r_callback = TensorboardCallback()

    # #Se empieza con el entrenamiento del agente
    # #random_agent(env=env)
    train_agent(env, time_steps = 500 * 50, input_callback = r_callback)
    # inference_agent(env, 3)

    #Se finaliza todo el setup del arduino
    env.reset()
    arduino_port.close()



