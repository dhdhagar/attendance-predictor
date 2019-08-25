# Importing the libraries
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# import matplotlib.pyplot as plt

# Declare constants
MAX_PERIOD_LEN = 45


def read_prep_data(filename):
    file_ext = filename.split(".")[-1]
    switcher = {
        "dta": pd.read_stata
    }
    try:
        df = switcher.get(file_ext)(filename)
        # Add a age as a constantly varying attribute even in one session
        df['age_days'] = df['date'] - df['dob']
        mmsc = MinMaxScaler()
        df['age_days'] = mmsc.fit_transform(df.iloc[:, -1:])
        # Group data by each session for every student
        grouped = df.groupby(['cohort', 'session', 'id'])
        # Setting the time-steps as the maximum length of a menstrual cycle
        timesteps = MAX_PERIOD_LEN
        x_input, y_output = [], []
        # Each row in X_train contains the gender, age, and 45-day absence history for each student
        # Each row in y_train contains the actual absence value for the 46th day for each 45-day historical set
        for name, group in grouped:
            group_demo = np.squeeze(np.array(group.loc[:, ['girl', 'age_days']].head(1)))
            n_series = group.shape[0] - timesteps
            if n_series > 0:
                for i in range(timesteps, group.shape[0]):
                    sys.stdout.write("\rProgress: %s: %i / %i" % (name, i - timesteps + 1, n_series))
                    sys.stdout.flush()
                    x_input.append(np.append(group_demo, np.array(group['entry'][i - timesteps:i]), axis=0))
                    y_output.append(group['entry'][i:i + 1].values[0])
        x_input, y_output = np.array(x_input), np.array(y_output)
        # Reshape X_train to make it consumable by keras (and to add more time-dependent predictors later)
        x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))
        return x_input, y_output
    except TypeError:
        print("ERROR: %s input file extension not suitable" % file_ext)
        exit(1)


def train_rnn(x_train, y_test):
    # Initialising the RNN
    rnn = Sequential()
    # Adding LSTM layers and dropout regularization
    # Units decided through the formula Nh = Ns / (alpha * (Ni + No))
    # => Nh = 365269 / (2 * (47 + 1)) = 3,804.88 ~ 3800
    rnn.add(LSTM(units=3800, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    rnn.add(LSTM(units=3800))
    rnn.add(Dropout(0.2))
    # Adding the output layer
    rnn.add(Dense(units=1, activation='sigmoid'))
    # Compiling the RNN
    rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(rnn.summary())
    # Fitting the RNN to the Training set
    rnn.fit(x_train, y_train, epochs=10, batch_size=64)
    return rnn


if __name__ == "__main__":
    X_train, y_train = read_prep_data("data/attendance_subset_2007-9.dta")
    model = train_rnn(X_train, y_train)