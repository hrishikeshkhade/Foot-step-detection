import RPi.GPIO as GPIO
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Define GPIO pins for seismic sensors
SENSOR_PIN_1 = 17
SENSOR_PIN_2 = 18

# Initialize GPIO settings
GPIO.setmode(GPIO.BCM)
GPIO.setup(SENSOR_PIN_1, GPIO.IN)
GPIO.setup(SENSOR_PIN_2, GPIO.IN)

# Global variables for LSTM model
SEQUENCE_LENGTH = 10
FEATURES = 2  # Two sensors
model = Sequential()
model.add(LSTM(50, input_shape=(SEQUENCE_LENGTH, FEATURES)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create a scaler for normalizing sensor data
scaler = MinMaxScaler()

# Initialize variables for storing sensor data sequences
sensor_data_sequence_1 = np.zeros((SEQUENCE_LENGTH, FEATURES))
sensor_data_sequence_2 = np.zeros((SEQUENCE_LENGTH, FEATURES))

def detect_footsteps(sensor_data, sensor_sequence):
    global model

    # Add new sensor data to the sequence
    sensor_sequence[:-1] = sensor_sequence[1:]
    sensor_sequence[-1] = sensor_data

    # Normalize the data
    normalized_data = scaler.transform(sensor_sequence.reshape(1, -1))

    # Reshape the data for the LSTM model
    reshaped_data = normalized_data.reshape(1, SEQUENCE_LENGTH, FEATURES)

    # Predict using the LSTM model
    prediction = model.predict(reshaped_data)

    # Threshold for detecting footsteps (you may need to adjust this)
    threshold = 0.5

    return prediction > threshold

try:
    while True:
        # Read sensor data
        sensor_1_data = GPIO.input(SENSOR_PIN_1)
        sensor_2_data = GPIO.input(SENSOR_PIN_2)

        # Update sensor data sequences
        detection_1 = detect_footsteps(sensor_1_data, sensor_data_sequence_1)
        detection_2 = detect_footsteps(sensor_2_data, sensor_data_sequence_2)

        if detection_1:
            print("Footstep detected by Sensor 1")

        if detection_2:
            print("Footstep detected by Sensor 2")

        time.sleep(0.1)  # Adjust this value based on your sensor's sampling rate

except KeyboardInterrupt:
    # Cleanup GPIO settings on program exit
    GPIO.cleanup()
