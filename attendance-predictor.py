# Importing the libraries
import math
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from sklearn.metrics import confusion_matrix

# import matplotlib.pyplot as plt

# Declare constants
MAX_PERIOD_LEN = 45


def read_prep_data(filename, max_session=math.inf):
    file_ext = filename.split(".")[-1]
    switcher = {
        "dta": pd.read_stata,
        "csv": pd.read_csv
    }
    try:
        df = switcher.get(file_ext)(filename)
        # Purge rows where any of the columns of interest have null values
        cols_of_interest = ['cohort', 'session', 'id', 'girl', 'date', 'dob', 'total_ab_sess', 'entry']
        df = df[df.loc[:, cols_of_interest].notnull().all(axis=1)]
        # Filter rows based on the maxsession argument (default: infinity)
        df = df.loc[(df['session'] <= max_session)]
        # Convert dob and date columns to date
        for datecol in ['dob', 'date']:
            if df[datecol].dtype == object:
                df[datecol] = pd.to_datetime(df[datecol])
        # Add age as a constantly varying attribute even in one session
        df['age_days'] = pd.to_numeric(df['date'] - df['dob'])
        mmsc = MinMaxScaler()
        df.loc[:, ['age_days', 'total_ab_sess']] = mmsc.fit_transform(df.loc[:, ['age_days', 'total_ab_sess']])
        # Group data by each session for every student
        grouped = df.groupby(['cohort', 'session', 'id'])
        # Setting the time-steps as the maximum length of a menstrual cycle
        timesteps = MAX_PERIOD_LEN
        x_input, y_output = [], []
        # Each row in X_train contains the gender, age, and 45-day absence history for each student
        # Each row in y_train contains the actual absence value for the 46th day for each 45-day historical set
        for name, group in grouped:
            group_demo = np.squeeze(np.array(group.loc[:, ['girl', 'age_days', 'total_ab_sess']].head(1)))
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


def train_rnn(x_input, y_output):
    # Initialising the RNN
    rnn = Sequential()
    # Adding LSTM layers and dropout regularization
    rnn.add(LSTM(units=100, return_sequences=True, input_shape=(x_input.shape[1], 1)))
    rnn.add(LSTM(units=100, return_sequences=True))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(units=100))
    rnn.add(Dropout(0.2))
    # Adding the output layer
    rnn.add(Dense(units=1, activation='sigmoid'))
    # Compiling the RNN
    rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(rnn.summary())
    # Fitting the RNN to the Training set
    rnn.fit(x_input, y_output, epochs=3, batch_size=64)
    return rnn


if __name__ == "__main__":
    # Training
    try:
        model = load_model('model/rnn.h5')
    except OSError:
        print("ERROR: Model file not found")
        X_train, y_train = read_prep_data("data/attendance_subset_2007-9.dta")
        model = train_rnn(X_train, y_train)
        model.save('model/rnn.h5')

    # Testing
    X_test, y_test = read_prep_data("data/chilla_2008_14_fixed.csv", max_session=2010)
    predicted_abs = model.predict(X_test)
    predicted_abs_thresh = np.squeeze(predicted_abs > (predicted_abs.max() / 2))

    # Making the Confusion Matrix
    # Columns represent predicted; Rows represent actual
    # eg. value in cell (1,0) represents the actual number of 1s predicted as 0s
    cm = confusion_matrix(y_test, predicted_abs_thresh)
    print("Confusion Matrix:")
    print(cm)
