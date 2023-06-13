import d3rlpy
import gym 
from gym_env_balancin_v2 import ControlEnv
import serial
import logging
import time
from gym.wrappers import NormalizeObservation, NormalizeReward
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import AWAC, CQL
from d3rlpy.wrappers.sb3 import SB3Wrapper
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import td_error_scorer
from sklearn.model_selection import train_test_split


## Cargar dataset de RandomPolicy
random_dataset = MDPDataset.load('C:/Users/B1500/OneDrive/Escritorio/repo_bicopter_RL/bicopter_RL_control/random_policy_dataset.h5')



# each episode is also splitted into d3rlpy.dataset.Transition objects
# episode = random_dataset.episodes[0]
# print(episode[0].observation)
# print(episode[0].action)
# print(episode[0].reward)
# print(episode[0].next_observation)
# print(episode[0].terminal)

# for episode in random_dataset.episodes[0:30]:
#     print(episode[0].reward)


#Se divide en train y test
train_episodes, test_episodes = train_test_split(random_dataset, test_size=0.2)


#Se instancia el algoritmo a usar 
cql = CQL()

#Se monta con el dataset
cql.build_with_dataset(random_dataset)


#Set metrics to use
# calculate metrics with test dataset
td_error = td_error_scorer(cql, test_episodes)


cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=60,
        scorers={
            'td_error': td_error_scorer,
            'value_scale': average_value_estimation_scorer,
        })


# save full parameters
cql.save_model('cql_v1.pt')



#Tranformando el modelo offline en un modelo sb3 para finetunning online
# d3rlpy model is accessible via `wrapped_model.algo`
# wrapped_model = SB3Wrapper(cql)
# wrapped_model.save(path="model_trained_from_offline_v00")