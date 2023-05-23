import serial
import time
import logging
import yaml
from gym.wrappers import FrameStack
import gym
import stable_baselines3
import tensorboard
import yaml
from gym_env_balancin import ControlEnv
from gym_env_balancin import TensorboardCallback
from stable_baselines3 import SAC


######### CODIGO PRUEBA DE ENTRENAMIENTO DE SOFT ACTOR-CRITIC ###########

def setup_arduino():
    """Funcion para hacer el septup del arduino"""
    arduino = serial.Serial('COM11', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
    print("Correctamente conectado")

    time.sleep(3)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    return arduino



def random_agent(env, episodes = 1):
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
            action, _ = agente.predict(observation, deterministic=True) # Tomar una acción aleatoria
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            print(action)
            print(observation)
            print(len(rewards))


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

    sac = SAC('MlpPolicy',
        env=env,
        batch_size= 256,
        learning_rate= 3e-4,
        buffer_size= 10000,
        ent_coef= 'auto',
        gamma= 0.99,
        tau = 0.01,
        train_freq= (5, 'episode'),
        gradient_steps= 1,
        learning_starts= 10000,
        use_sde=True,
        sde_sample_freq = 32,
        verbose=1,
        seed=42,
        tensorboard_log="./sac_testing_v0")

    sac.learn(total_timesteps = time_steps, callback = input_callback)

    sac.save(path="model_trained_v00")




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
    env = FrameStack(env,num_stack=5)

    logging.info("Estableciendo entorno de entrenamiento")
    time.sleep(2)
    env.reset()
    time.sleep(2)

    input("Vuelve a presionar enter para que el agente se ejecute",)

    # Se definen las variables a monitorizar en Tensorboard
    r_callback = TensorboardCallback()

    #Se empieza con el entrenamiento del agente
    #random_agent(env=env)
    train_agent(env, time_steps = 500 * 100, input_callback = r_callback)

    #Se finaliza todo el setup del arduino
    env.reset()
    arduino_port.close()



