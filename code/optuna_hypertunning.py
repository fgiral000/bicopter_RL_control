""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using A2C implementation from Stable-Baselines3
on a OpenAI Gym environment.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.

You can run this example as follows:
    $ python sb3_simple.py

"""
import pickle as pkl
from typing import Any, Dict
import serial
import gym
from gym.wrappers import NormalizeReward, NormalizeObservation
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from gym_env_balancin_v2 import ControlEnv, TensorboardCallback
import time
import logging
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances



def setup_arduino():
    """Funcion para hacer el septup del arduino"""
    arduino = serial.Serial('COM5', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
    print("Correctamente conectado")

    time.sleep(3)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    return arduino

arduino_port = setup_arduino()




N_TRIALS = 12
N_JOBS = 1
N_STARTUP_TRIALS = 4
N_EVALUATIONS = 2
N_TIMESTEPS = int(5e3)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 5
N_EVAL_ENVS = 1
TIMEOUT = int(60 * 90)  # 90 minutes


ENV_ID = ControlEnv(arduino_port)
ENV_ID = NormalizeObservation(ENV_ID)
ENV_ID = NormalizeReward(ENV_ID)

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
}





def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    learning_starts = trial.suggest_categorical("learning_starts", [100, 1000, 4000])
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = "auto"
    # You can comment that out when not using gSDE
    log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    activation_fn = trial.suggest_categorical('activation_fn', ["tanh", "relu", "gelu"])
    use_sde = True
    use_sde_at_warmup = False
    sde_sample_freq = trial.suggest_categorical('sde_sample_freq', [16, 128, 256, 512])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        # "large": [256, 256, 256],
        # "verybig": [512, 512, 512],
    }[net_arch]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
    }[activation_fn]

    target_entropy = "auto"
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_uniform('target_entropy', -10, 10)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "use_sde": use_sde,
        "use_sde_at_warmup": use_sde_at_warmup,
        "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch, activation_fn=activation_fn),
    }


    return hyperparams


def sample_tqc_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TQC hyperparams.

    :param trial:
    :return:
    """
    # TQC is SAC + Distributional RL
    hyperparams = sample_sac_params(trial)

    n_quantiles = trial.suggest_int("n_quantiles", 5, 50)
    top_quantiles_to_drop_per_net = trial.suggest_int("top_quantiles_to_drop_per_net", 0, n_quantiles - 1)

    hyperparams["policy_kwargs"].update({"n_quantiles": n_quantiles})
    hyperparams["top_quantiles_to_drop_per_net"] = top_quantiles_to_drop_per_net

    return hyperparams


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:

    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters
    kwargs.update(sample_tqc_params(trial))
    # Create the RL model
    model = TQC(**kwargs)
    # Create env used for evaluation
    eval_envs = ControlEnv(arduino_port)
    eval_envs = NormalizeObservation(eval_envs)
    eval_envs = NormalizeReward(eval_envs)
    # Create the callback that will periodically evaluate
    # and report the performance
    eval_callback = TrialEvalCallback(
        eval_envs,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    r_callback = TensorboardCallback()

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=[eval_callback,r_callback])
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_envs.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":


    #Se inicializa el entorno del arduino


    logging.info("Enciende la fuente de alimentacion")
    logging.info("Espera 20 segundos hasta que todo el sistema este activo")
    time.sleep(10)

    logging.info("El sistema se ha activado correctamente")

    input("Presiona la tecla enter cuando todo este preparado",)
    #Se establece el entorno de entrenamiento



    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")


    logging.info("Estableciendo entorno de entrenamiento")
    time.sleep(2)
    time.sleep(2)

    input("Vuelve a presionar enter para que el agente se ejecute",)


    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")

    with open("study.pkl", "wb+") as f:
        pkl.dump(study, f)

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()

    arduino_port.close()