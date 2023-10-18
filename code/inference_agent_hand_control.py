## Module to do inference with the trained agent and plot the results 
import serial
import threading
import time
import logging
from wrappers_from_rlzoo import ActionSmoothingWrapper, HistoryWrapper
from gym_env_balancin_v2 import ControlEnv
from stable_baselines3 import SAC
from sb3_contrib import TQC
import torch
import numpy as np
import cv2
import mediapipe as mp
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Configure Matplotlib to use LaTeX for text rendering
# matplotlib.rcParams['text.usetex'] = True

def setup_arduino():
    """Funcion para hacer el septup del arduino"""
    arduino = serial.Serial('/dev/ttyACM0', 9600) # Replace 'COM3' with the arduinoial port of your Arduino
    print("Correctamente conectado")

    time.sleep(3)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    return arduino


def hand_tracking_controller(env):
    # Initialize MediaPipe hand solution
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Capture video from the first camera device
    cap = cv2.VideoCapture(0)


    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1'
    fps = 30  # or whatever your source's frame rate is
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('../output.mp4', fourcc, fps, (frame_width, frame_height))


    def calculate_angle(point1, point2):
        """
        Calculate the angle between the line from point1 to point2 and the horizontal line from the frame center.
        """
        # Calculate the angle of the line with the y-axis
        angle = math.degrees(math.atan2(point1.y - point2.y, point1.x - point2.x))

        # You might need to adjust the returned angle here depending on how you wish to measure the angles
        return angle

    while cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get the hand landmarks
        results = hands.process(rgb_image)

        # Draw the hand annotations on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Assuming you wish to measure the angle of the line from the wrist (landmark 0) to the middle finger MCP joint (landmark 9)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                # Calculate the angle
                angle = calculate_angle(wrist, middle_finger_mcp) - 90
                # print(angle)
                # Set theta_referencia based on the angle
                if angle > 20:
                    new_theta_value = 25.0
                elif angle < -20:
                    new_theta_value = -25.0
                else:
                    new_theta_value = 0

                
                env.env_method("set_theta_reference", new_theta_value)
                # Here, integrate the code to send theta_referencia to your system
                # For example, you might use a function like this (you'll need to define it according to your setup):
                # set_theta_reference(theta_referencia)
        else:
            env.env_method("set_theta_reference", 0.0)
        # Display the frame
        cv2.imshow('MediaPipe Hands', frame)

        # Write the frame to the video file
        out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def animate(i, x, y1, y2, line1, line2):
    line1.set_data(x[:i], y1[:i])
    line2.set_data(x[:i], y2[:i])
    return line1, line2

def create_animation(x, y1, y2):
    fig, ax = plt.subplots()
    # Updated legends with LaTeX formatting
    # line1, = ax.plot([], [], 'b-', label=r'Current Angle ($\theta$)')
    # line2, = ax.plot([], [], 'r-', label=r'Reference Angle ($\theta_{\mathrm{ref}}$)')
    line1, = ax.plot([], [], 'b-', label=r'Current Theta')
    line2, = ax.plot([], [], 'r-', label=r'Reference Theta')

    ax.set_xlim(min(x), max(x))
    ax.set_ylim(-60, 60)  # Adjust if your data range is different

    # Labels for the axes with LaTeX formatting
    ax.set_xlabel(r'Current Time Step')
    ax.set_ylabel(r'Theta Value ($^\circ$)')

    # Legend for the plot
    ax.legend(loc='upper right')

    ani = animation.FuncAnimation(
        fig, animate, len(x), fargs=[x, y1, y2, line1, line2],
        interval=25, blit=True
    )

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='FranGiral'), bitrate=1800)

    # Save the animation as an MP4 file
    ani.save('../bicopter_control.mp4', writer=writer)

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

    # Initializing the Arduino environment
    arduino_port = setup_arduino()
    logging.info("Turn on the power supply")
    logging.info("Wait 10 seconds until the entire system is active")
    time.sleep(10)
    logging.info("The system has been activated correctly")
    input("Press the enter key when everything is ready",)

    # Setting up the training environment
    env = ControlEnv(arduino_port)
    env = Monitor(env)
    # env = TimeLimit(env, max_episode_steps=200)
    env = ActionSmoothingWrapper(env, smoothing_coef=0.6)
    env = HistoryWrapper(env=env)
    # VecNormalize wrappers
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(load_path=VEC_ENV_NEW, venv=env)
    env.training = False
    env.norm_reward = False
    logging.info("Setting up training environment")
    input("Press enter again to execute the agent",)





    #Load the pre-trained agent
    model = TQC.load(MODEL_NAME_NEW)
    model.set_env(env)
    model.set_parameters(MODEL_NAME_NEW)


    ############################################ HANDS CONTROL #########################################################3

    hand_tracking_thread = threading.Thread(target=hand_tracking_controller, args=(env,), daemon=True)
    hand_tracking_thread.start()
    ####################################################################################################################


    #Using stable-baselines3 to use the pre-trained policy in inference
    observation = env.reset()
    data_storage = []
    current_step = 0
    logging.info("The current reference angle is:",)
    # print(rt)
    
    while True:

        try:
            current_step +=1
            action, _ = model.predict(observation, deterministic=True)
            observation, rewards, done, info = env.step(action)
            unnorm_observation = env.get_original_obs()

            #################################################### plotting ##########################################################
            current_angle = unnorm_observation[0][16]
            reference_angle = unnorm_observation[0][18]
            data_storage.append((current_step, current_angle, reference_angle))
            #############################################################################################################################


        except KeyboardInterrupt:
            print("The process has been finalized by keyboard command")
            #Arduino setup and serial port are closed
            arduino_port.close()
            # Post-process the collected data to create an animation
            x, y1, y2 = zip(*data_storage)  # Unpack the data
            create_animation(x, y1, y2)
            break

