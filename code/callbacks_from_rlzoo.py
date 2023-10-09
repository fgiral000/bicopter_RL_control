import os
import pickle
import tempfile
import time
from copy import deepcopy
from functools import wraps
from threading import Thread
from typing import Optional

# import optuna
from sb3_contrib import TQC
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization




class ParallelTrainCallback(BaseCallback):
    """
    Callback to explore (collect experience) and train (do gradient steps)
    at the same time using two separate threads.
    Normally used with off-policy algorithms and `train_freq=(1, "episode")`.
    TODO:
    - blocking mode: wait for the model to finish updating the policy before collecting new experience
        at the end of a rollout
    - force sync mode: stop training to update to the latest policy for collecting
        new experience
    :param gradient_steps: Number of gradient steps to do before
        sending the new policy
    :param verbose: Verbosity level
    :param sleep_time: Limit the fps in the thread collecting experience.
    """

    def __init__(self, gradient_steps: int = 100, verbose: int = 0, sleep_time: float = 0.0):
        super(ParallelTrainCallback, self).__init__(verbose)
        self.batch_size = 0
        self._model_ready = True
        self._model = None
        self.gradient_steps = gradient_steps
        self.process = None
        self.model_class = None
        self.sleep_time = sleep_time

    def _init_callback(self) -> None:
        temp_file = tempfile.TemporaryFile()

        # Windows TemporaryFile is not a io Buffer
        # we save the model in the logs/ folder
        if os.name == "nt":
            temp_file = os.path.join("logs", "model_tmp.zip")

        self.model.save(temp_file)

        if self.model.get_vec_normalize_env() is not None:
            temp_file_norm = os.path.join("../logs", "vec_normalize.pkl")

            with open(temp_file_norm, "wb") as file_handler:
                pickle.dump(self.model.get_vec_normalize_env(), file_handler)

        # TODO: add support for other algorithms
        for model_class in [SAC, TQC]:
            if isinstance(self.model, model_class):
                self.model_class = model_class
                break

        assert self.model_class is not None, f"{self.model} is not supported for parallel training"
        self._model = self.model_class.load(temp_file)

        if self.model.get_vec_normalize_env() is not None:
            with open(temp_file_norm, "rb") as file_handler:
                self._model._vec_normalize_env = pickle.load(file_handler)
                self._model._vec_normalize_env.training = False

        self.batch_size = self._model.batch_size

        # Disable train method
        def patch_train(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                return

            return wrapper

        # Add logger for parallel training
        self._model.set_logger(self.model.logger)
        self.model.train = patch_train(self.model.train)

        # Hack: Re-add correct values at save time
        def patch_save(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                return self._model.save(*args, **kwargs)

            return wrapper

        self.model.save = patch_save(self.model.save)

    def train(self) -> None:
        self._model_ready = False

        self.process = Thread(target=self._train_thread, daemon=True)
        self.process.start()

    def _train_thread(self) -> None:
        self._model.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
        self._model_ready = True

    def _on_step(self) -> bool:
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        return True

    def _on_rollout_end(self) -> None:
        if self._model_ready:
            self._model.replay_buffer = deepcopy(self.model.replay_buffer)
            self.model.set_parameters(deepcopy(self._model.get_parameters()))
            self.model.actor = self.model.policy.actor
            # Sync VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                sync_envs_normalization(self.model.get_vec_normalize_env(), self._model._vec_normalize_env)

            if self.num_timesteps >= self._model.learning_starts:
                self.train()
            # Do not wait for the training loop to finish
            # self.process.join()

    def _on_training_end(self) -> None:
        # Wait for the thread to terminate
        if self.process is not None:
            if self.verbose > 0:
                print("Waiting for training thread to terminate")
            self.process.join()




# class LapTimeCallback(BaseCallback):
#     def _on_training_start(self):
#         self.n_laps = 0
#         output_formats = self.logger.output_formats
#         # Save reference to tensorboard formatter object
#         # note: the failure case (not formatter found) is not handled here, should be done with try/except.
#         self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

#     def _on_step(self) -> bool:
#         lap_count = self.locals["infos"][0]["lap_count"]
#         lap_time = self.locals["infos"][0]["last_lap_time"]

#         if lap_count != self.n_laps and lap_time > 0:
#             self.n_laps = lap_count
#             self.tb_formatter.writer.add_scalar("time/lap_time", lap_time, self.num_timesteps)
#             if lap_count == 1:
#                 self.tb_formatter.writer.add_scalar("time/first_lap_time", lap_time, self.num_timesteps)
#             else:
#                 self.tb_formatter.writer.add_scalar("time/second_lap_time", lap_time, self.num_timesteps)
#             self.tb_formatter.writer.flush()