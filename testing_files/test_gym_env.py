import gym 
from gym_env_balancin import ControlEnv
from stable_baselines3.common.env_checker import check_env
import serial

arduino = serial.Serial('COM5', 9600)
env = ControlEnv(arduino)

check_env(env)

print("**************TESTEANDO EL ENTORNO DE GYM ************************")
print("Dimension del espacio de acciones:", env.action_space.shape)
print("Dimension del espacio de estados:" , env.observation_space._shape)

print("**************EJEMPLOS DE LOS ESPACIOS DE ACCIONES Y ESTADOS ************************")
print("Ejemplo del espacio de acciones:", env.action_space.sample())
print("Ejemplo del espacio de estados:" , env.observation_space.sample())
print("Ejemplo del espacio de acciones:", env.action_space.dtype)
print("Ejemplo del espacio de estados:" , env.observation_space.sample())