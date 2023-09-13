# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
#import tensorflow as tf
import tensorboard
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
import time

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.primary_reward = 0
        self.secondary_reward = 0
        self.log_path = './sac_testing_v5'

    def _on_rollout_end(self) -> None:
        pass


    def _on_step(self) -> bool:

        # Sacar los datos del env
        # self.primary_reward = self.training_env.get_attr('reward_1')[0]
        # self.time_reward = self.training_env.get_attr('reward_2')[0]
        # self.goal_reward = self.training_env.get_attr('reward_3')[0]
        # self.timeout_reward = self.training_env.get_attr('reward_4')[0]

        self.space_state = self.training_env.get_attr('current_state')[0]
        self.action = self.training_env.get_attr('last_action')[0]
        self.max_theta = self.training_env.get_attr('max_theta')[0]
        self.max_velocity = self.training_env.get_attr('max_velocity')[0]
        # current_step = self.training_env.get_attr('current_step')[0]

        # Espacio de los estados
        state_space = np.array(self.space_state)

        theta = state_space[0]
        self.logger.record("state_space/theta", theta)

        # theta_denorm = theta*self.max_theta
        # self.logger.record("state_space/theta_denorm", theta_denorm)


        theta_dot = state_space[1]
        self.logger.record("state_space/theta_dot", theta_dot)

        # theta_dot_denorm = theta_dot*self.max_velocity
        # self.logger.record("state_space/theta_dot_denorm", theta_dot_denorm)


        # Espacio de acciones

        actions = np.array(self.action)
        actions_denorm = self.denormalize_action(actions)

        Ti = actions[0]
        self.logger.record("action_space/Left_Thrust", Ti)

        Ti_denorm = actions_denorm[0]
        self.logger.record("action_space/Left_Thrust_Denorm", Ti_denorm)

        Td = actions[1]
        self.logger.record("action_space/Right_Thrust", Td)

        Td_denorm = actions_denorm[1]
        self.logger.record("action_space/Right_Thrust_Denorm", Td_denorm)

        # Rewards
        # self.total_reward = self.primary_reward + self.time_reward + self.goal_reward + self.timeout_reward

        # self.logger.record("rewards/total_reward", self.total_reward)

        # self.logger.record("rewards/primary_reward", self.primary_reward)

        # self.logger.record("rewards/time_reward", self.time_reward)

        # self.logger.record("rewards/goal_reward", self.goal_reward)

        # self.logger.record("rewards/timeout_reward", self.timeout_reward)


        # Imprimo todo
        self.logger.dump(self.num_timesteps)

        return True


    def denormalize_action(self, action):
        return ( (action + 1) * (300/2) ) + 1000


class ControlEnv(gym.Env):

    def __init__(self, arduino_port):
        super(ControlEnv, self).__init__()
        # Espacio de acciones continuas entre 1000 y 1500
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # Espacio de estados de dimensión 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        # Frecuencia de acciones de 50Hz
        self.action_freq = 50
        # Time out de 200 time steps
        self.max_steps = 500

        # Máximo reward
        self.max_reward = 0.0
        # Early stopping después de 10 time steps con el máximo reward
        self.max_reward_steps = 250
        self.current_reward_steps = 0

        self.counter = 0            # Numero de steps consecutivos dentro de la zona buena

        self.max_angle_steps = 0
        ###Valores maximos de los parametros del vector de estado
        self.max_theta = 25
        self.max_velocity = 150


        #Valores del vector de estados
        ##############################################################################
        #AQUI SE DEBEN INTRODUCIR LOS VALORES LEIDOS DEL ARDUINO ANTES DE EMPEZAR EL EPISODIO
        ###################################################################################3
        self.theta_referencia = 0.0
        self.theta_inicial = 0.0
        self.theta_velocity_inicial = None
        self.current_step = 0

        # Estado actual
        self.current_state = np.array([self.theta_inicial, self.theta_velocity_inicial, self.theta_referencia, (self.theta_inicial - self.theta_referencia)**2], dtype=np.float32)


        self.last_action = None
        self.arduino_values = None

        self.previous_shaping = None
        self.arduino_port = arduino_port
        self.arduino_port.reset_input_buffer()

        self.render_mode = None
        # Inicialización
        self.reset()



    def reset(self,seed=42):
        # Reiniciar el estado y el contador de time steps y reward steps

        self.current_step = 0
        self.current_reward_steps = 0
        self.max_angle_steps = 0

        self.last_action = np.array([-1,-1], dtype=np.float32)
        self.filtered_action_past = self.last_action

        self.send_action(np.array([-1,-1], dtype=np.float32), self.arduino_port)
        time.sleep(2)
        #Se deben reiniciar tambien los valores de estado iniciales
        self.arduino_values = self.get_observation(self.arduino_port)
        ##############################################################################
        #AQUI SE DEBEN INTRODUCIR LOS VALORES LEIDOS DEL ARDUINO ANTES DE EMPEZAR EL EPISODIO
        ###################################################################################3
        # Valores del vector de estados
        self.theta_referencia = 0.0
        self.theta_inicial = self.arduino_values[0] 
        self.theta_velocity_inicial = self.arduino_values[1] 
        self.current_step = 0

        self.previous_shaping = None
        self.current_state = np.array([self.theta_inicial, self.theta_velocity_inicial, self.theta_referencia, (self.theta_inicial - self.theta_referencia)**2], dtype=np.float32)

        # self.current_state = np.array(self.current_state, dtype=np.float32)

        self.arduino_port.reset_input_buffer()
        self.arduino_port.reset_output_buffer()
        ##########################################################################################################
        #####TAMBIEN HAY QUE PONER LOS MOTORES A 0 CUANDO SE ACABA EL EPISODIO, PERO SIN NECESIDAD DE APAGAR LA CORRIENTE
        ############################################################################################################

        
        info = {}
        # Devolver el estado actual
        return self.current_state, info



    def step(self, action):

        ####### ENVIAR ACCCION TOMADA AL ARDUINO ########

        # filtered_action = 0.8 * self.filtered_action_past + 0.2 * action

        self.send_action(action, self.arduino_port)


        # Incrementar el contador de time steps
        self.current_step += 1


        # Calcular el nuevo estado a partir de la acción y el estado actual
        ##### CALCULAR ESTADO DESDE EL ARDUINO ########
        new_arduino_data = self.get_observation(self.arduino_port)
        ########################################################################################
        # AQUI SE DEBEN INTRODUCIR LOS VALORES LEIDOS DEL ARDUINO ANTES DE EMPEZAR EL EPISODIO #
        ########################################################################################

        new_state = np.array([new_arduino_data[0] , new_arduino_data[1], self.theta_referencia, (new_arduino_data[0] - self.theta_referencia)**2], dtype=np.float32)

        # ---------------------------- RECOMPENSAS -------------------------

        # reward = - abs(new_state[0] - new_state[2])

        # reward = (

        #             (abs(new_state[0] - new_state[2]))

        #             -(self.current_step / self.max_steps) * (abs(new_state[0] - new_state[2]) > (3/self.max_theta))

        #             + 2 * (abs(new_state[0] - new_state[2]) <= (3/self.max_theta)) 

        #         )
        
        # if self.previous_shaping is not None:

        #     reward = shaping - self.previous_shaping

        # self.previous_shaping = shaping
        # -------------------------------------------------------------------
        # -------------------------------------------------------------------

        reward = 0

        if abs(new_state[0] - new_state[2]) <= (10):
            reward = 1 - (abs(new_state[0] - new_state[2]) / self.max_theta)

       # reward = 0.8 * reward - 0.1 * abs(action[0] - self.last_action[0]) - 0.1 * abs(action[1] - self.last_action[1])
        
        # if abs(new_state[1]) <= (80/self.max_velocity):
        #     reward+=0.5


        # -------------------------------------------------------------------

        #Se guarda la ultima accion tomada para utilizarla en el reward
        # self.filtered_action_past = filtered_action
        self.last_action = action

        # Actualizar el contador de reward steps si se alcanza el máximo reward
        # if abs(new_state[0] - new_state[2]) <= (3):
        #     self.current_reward_steps += 1
        # else:
        #     self.current_reward_steps = 0

               
        if abs(new_state[0] - new_state[2]) >= (20):
            self.max_angle_steps += 1
        else:
            self.max_angle_steps = 0

        # Comprobar si se ha alcanzado el time out o el early stopping
        if self.current_reward_steps == self.max_reward_steps:
            # reward+=100
            # done = True
            pass

        # elif self.current_step == self.max_steps:
            #reward-=1000
            # done = True

        elif self.max_angle_steps == 20:
            reward-=100
            done = True

        else:
            done = False

        terminated = False
        truncated = False

        if done:
            truncated = True


        # Actualizar el estado actual
        self.current_state = new_state

        # Devolver el nuevo estado, la recompensa, si el episodio ha terminado y un diccionario vacío de información adicional
        return self.current_state, reward, terminated,truncated, {}



    def _calculate_reward(self, state):
        # Calcular la recompensa como la suma de los componentes del estado
        return np.clip((- (state[0] - state[2]) ** 2 + 9/625)*625*10/9, -20, 20)

    def _time_reward(self):
        #  Cada paso que des quitas 1 de recompensa, para incentivar que se de prisa
        return -1

    def _goal_reached(self):

        if self.reward_1 >= 0:
            self.counter+=1
        else:
            self.counter = 0

        if self.counter >= 100:
            reward = 100
            done = True
            self.counter = 0

        else:
            reward = 0
            done = False

        return reward, done


    def _timeout(self):
        if self.current_step >= self.max_steps:
            done = True
            reward = -100
        else:
            done = False
            reward = 0

        return reward, done


    def _secondary_reward(self, action, last_action):
        return - ( ((action[0] - last_action[0]) ** 2) + ((action[1] - last_action[1]) ** 2) )


    def get_observation(self, arduino):
        """Metodo para recibir informacion del estado desde el arduino"""

        # Wait until the start character '<' is received
        while True:
            if arduino.read().decode() == '<':
                break

        # Read the data until the end character '>' is received
        data = ''
        while True:
            char = arduino.read().decode()
            if char == '>':
                arduino.write('?'.encode())
                break
            data += char

        # Convert the data string to a list of integers
        data_list = list(map(float, data.split(',')))
        return data_list



    def send_action(self, action, arduino):
        """método para enviar la accion tomada por el agente hacia el arduino"""
        "Hay que de-normalizar las acciones"
        action_complete = self.denormalize_action(action)

        actions = np.round(action_complete, decimals=1)
        # Convert sequence to string
        sequenceString = '$' + str(actions[0]) + str(actions[1]) + '#' # Assumes sequence is a 2D array with shape (1, 2)
        # Send sequence to Arduino
        arduino.write(sequenceString.encode())


    def denormalize_action(self, action):
        return ( (action + 1) * (300/2) ) + 1000
