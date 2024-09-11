# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:50:41 2024

@author: rsadeghianbroujeny
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:07:10 2024

@author: rsadeghianbroujeny
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:35:14 2024

@author: rsadeghianbroujeny
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:06:50 2024

@author: rsadeghianbroujeny
"""

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1DTranspose, UpSampling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dropout, RepeatVector, TimeDistributed
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import itertools
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.ndimage import gaussian_filter
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# Set paths and load data
path = r"C:/Users/rsadeghianbroujeny/OneDrive - Cesi/CESI/Machine virtuelle/cmapss"
# path=r"C:\Users\2411609\OneDrive - Cesi\CESI\Machine virtuelle\cmapss"
# path=r"C:\Users\ArRARV\OneDrive - Cesi\CESI\Machine virtuelle\cmapss"

os.chdir(path)
x1='my_model_convlstm%s.h5'
x2='my_model_convmlp%s.h5'
N=0
filename1 = 'train_FD004.txt'
filename2 = 'test_FD004.txt'
filename3 = 'RUL_FD004.txt'

train = np.genfromtxt(filename1, delimiter=' ')
test = np.genfromtxt(filename2, delimiter=' ')
RULt = np.genfromtxt(filename3, delimiter=' ')

# Creating DataFrames
columns = ['engine_id', 'time'] + [f'sensor{i}' for i in range(1, train.shape[1] - 1)]
train_data = pd.DataFrame(train, columns=columns)
test_data = pd.DataFrame(test, columns=columns)
RUL_data = pd.DataFrame(RULt, columns=['RUL'])

# Drop specified sensors
sensors_to_drop = ['sensor1','sensor2','sensor3','sensor4','sensor5', 'sensor8','sensor9','sensor11','sensor19','sensor21','sensor22']
train_data = train_data.drop(columns=sensors_to_drop)
test_data = test_data.drop(columns=sensors_to_drop)

# Calculate RUL for training set
max_cycles_train = train_data.groupby('engine_id')['time'].transform(max)
train_data['RUL'] = max_cycles_train - train_data['time']

# Calculate RUL for test set
max_cycles_test = test_data.groupby('engine_id')['time'].transform(max)
test_data['RUL'] = max_cycles_test - test_data['time']
RUL_data.index = np.arange(1, len(RUL_data) + 1)
# Dictionary to store RUL values for each engine_id
rul_dict = {}

# Iterate over unique engine_ids in test_data
for engine_id in test_data['engine_id'].unique():
    # Find the first instance of RUL for this engine_id
    rul_value = RUL_data.loc[engine_id, 'RUL']
    
    # Store this RUL value in the dictionary
    rul_dict[engine_id] = rul_value

# Update RUL values in test_data
for engine_id, rul_value in rul_dict.items():
    # Find rows with the current engine_id and update RUL
    test_data.loc[test_data['engine_id'] == engine_id, 'RUL'] += rul_value

# Define the piecewise function
def apply_piecewise_rul(rul, threshold=125):
    return np.minimum(rul, threshold)

# Apply piecewise function to train_data
train_data['RUL'] = train_data['RUL'].apply(apply_piecewise_rul)

# Apply piecewise function to test_data
test_data['RUL'] = test_data['RUL'].apply(apply_piecewise_rul)

# For RUL_data (if it contains the final RUL values for the test set)
RUL_data['RUL'] = RUL_data['RUL'].apply(apply_piecewise_rul)

# Verify the changes
print(train_data[['engine_id', 'time', 'RUL']].head())
print(test_data[['engine_id', 'time', 'RUL']].head())
print(RUL_data.head())



# Extract features and target
features_train= train_data.drop([ 'engine_id','time'], axis=1).values

# Extract features and target
features_test= test_data.drop(['engine_id','time'], axis=1).values

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Extract features and target
features_train= train_data.drop([ 'time'], axis=1).values

# Extract features and target
features_test= test_data.drop(['time'], axis=1).values

from sklearn.preprocessing import MinMaxScaler
val_min_train=np.empty([1,14])
val_max_train=np.empty([1,14])
val_min_test=np.empty([1,14])
val_max_test=np.empty([1,14])
feature_data_train_normalized=np.copy(features_train)
feature_data_test_normalized=np.copy(features_test)

for j3 in range(1,15):

        
        val_min_train[:,j3-1]=np.min(features_train[:,j3])
        val_max_train[:,j3-1]=np.max(features_train[:,j3])
        val_min_test[:,j3-1]=np.min(features_test[:,j3])
        val_max_test[:,j3-1]=np.max(features_test[:,j3])
        feature_data_train_normalized[:,j3]=(features_train[:,j3]-np.min(features_train[:,j3])) / (np.max(features_train[:,j3])-np.min(features_train[:,j3]))
    
    
    
        feature_data_test_normalized[:,j3] = (features_test[:,j3] - np.min(features_train[:,j3])) / (np.max(features_train[:,j3])-np.min(features_train[:,j3]))



# Extract features and target
target_train_normalized = feature_data_train_normalized[:,14]
updated_matrix = np.column_stack((target_train_normalized, features_train[:,0]))

feature_data_train_normalized= feature_data_train_normalized[:,1:14]

# Extract features and target
target_test_normalized = feature_data_test_normalized[:,14]
updated_matrix_test = np.column_stack((target_test_normalized, features_test[:,0]))

feature_data_test_normalized= feature_data_test_normalized[:,1:14]
RUL_data_normalized=(RUL_data - np.min(features_train[:,j3])) / (np.max(features_train[:,j3])-np.min(features_train[:,j3]))

# Generate numbers from 1 to the length of RUL_data_normalized
second_column_values = np.arange(1, len(RUL_data_normalized) + 1)

# Create a 2D array with RUL_data_normalized as the first column and second_column_values as the second column
RUL_data_normalized = np.column_stack((RUL_data_normalized, second_column_values))

def create_conv_lstm():
    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(filters=256, kernel_size=1, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=1))
    # model.add(LSTM(256, activation='relu', return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    return model



def create_conv_mlp():
    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=1, activation='relu', input_shape=(X_train.shape[1],  X_train.shape[2])))
    model.add(Conv1D(filters=256, kernel_size=1, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    return model

# Define the sliding window function

def create_sliding_windows(features, target, window_size):
    X = []
    y = []
    ids = []
    for i in range(len(features) - window_size + 1):
        X.append(features[i:i+window_size])
        y.append(target[i+window_size-1])
    return np.array(X), np.array(y)

# def create_sliding_windows(features, targets, window_size):
#     X = []
#     y = []
#     for i in range(len(features) - window_size):
#         X.append(features[i:i+window_size])
#         y.append(targets[i+window_size])
#     return np.array(X), np.array(y)


delays=30
n_training=1
error_rmse_convlstm=np.empty([delays,n_training])
error_rmse_convmlp=np.empty([delays,n_training])
error_rmse_meta=np.empty([delays,n_training])

error_mae_convlstm=np.empty([delays,n_training])
error_mae_convmlp=np.empty([delays,n_training])
error_mae_meta=np.empty([delays,n_training])

r2_convlstm=np.empty([delays,n_training])
r2_convmlp=np.empty([delays,n_training])
r2_meta=np.empty([delays,n_training])

for i in range(1,delays+1):
    window_size = i  # Adjust according to your requirement

    # Initialize lists to store the results
    X_train = []
    y_train = []
    
    # Assuming feature_data_train_normalized and target_train_normalized are already numpy arrays
    engine_ids = features_train[:, 0]  # Assuming engine IDs are in the first column of features
    
    # Iterate over unique engine IDs
    for engine_id in np.unique(engine_ids):
        # Mask to select data for the current engine
        engine_mask = (engine_ids == engine_id)
        
        # Select features and targets for the current engine
        engine_features = feature_data_train_normalized[engine_mask, 1:]  # Assuming first column is engine ID
        engine_targets = updated_matrix[engine_mask]
        
        # Apply sliding windows to the engine data
        X_engine, y_engine = create_sliding_windows(engine_features, engine_targets, window_size)
        
        # Append the results for the current engine to the lists
        X_train.append(X_engine)
        y_train.append(y_engine)
    
    # Convert lists to numpy arrays
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train ,shuffle=False, test_size = 0.2)

    # Print the shapes of the resulting arrays
    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)
        
    
        
        
    # Initialize lists to store the results
    X_test = []
    y_test = []
    
    # Assuming feature_data_test_normalized and target_test_normalized are already numpy arrays
    engine_ids_test = features_test[:, 0]  # Assuming engine IDs are in the first column of features
    
    # Iterate over unique engine IDs in test data
    for engine_id in np.unique(engine_ids_test):
        # Mask to select data for the current engine
        engine_mask = (engine_ids_test == engine_id)
        
        # Select features and targets for the current engine
        engine_features = feature_data_test_normalized[engine_mask, 1:]  # Assuming first column is engine ID
        engine_targets = updated_matrix_test[engine_mask]
        
        # Apply sliding windows to the engine data
        X_engine, y_engine = create_sliding_windows(engine_features, engine_targets, window_size)
    
        
        # Append the results for the current engine to the lists, if X_engine is not empty
        if len(X_engine) > 0:
            X_test.append(X_engine)
            y_test.append(y_engine)
    
    # Convert the lists of arrays into single numpy arrays
    if X_test:
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
    else:
        X_test = np.array([])  # Handle case where X_test is empty
        y_test = np.array([])  # Handle case where y_test is empty
    
    # Print the shapes of the resulting arrays
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)
    import numpy as np
    
    # Assuming the engine ID is in the second column of y_test
    engine_ids = np.unique(y_test[:, 1])  # Assuming engine ID is in the second column
    
    # Initialize lists to store the last instances
    last_instances_X = []
    last_instances_y = []
    
    # Iterate over each unique engine ID
    for engine_id in engine_ids:
        # Find the indices where engine_id occurs in the second column of y_test
        indices = np.where(y_test[:, 1] == engine_id)[0]
        
        # Get the last index for this engine_id
        last_index = indices[-1]
        
        # Extract the corresponding X_test and y_test values
        last_instances_X.append(X_test[last_index])
        last_instances_y.append(y_test[last_index])
    
    # Convert lists to numpy arrays
    last_instances_X = np.array(last_instances_X)
    last_instances_y = np.array(last_instances_y)
    last_instances_y=last_instances_y[:,0]
    # Now last_instances_X contains the last instances of X_test and last_instances_y contains the corresponding y_test values.
    
    
    
    
    
    y_test=y_test[:,0]
    
    
    
    
    y_train=y_train[:,0]
    # y_test=y_test[:,0]
    
    y_val=y_val[:,0]
    n_features=X_train.shape[2]
    for k in range(0,n_training):
        N=N+1

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        
        conv_lstm_model = create_conv_lstm()
        conv_mlp_model = create_conv_mlp()
        
        conv_lstm_model.fit(X_train, y_train, epochs=30, batch_size=4000, callbacks=[early_stopping], validation_data=(X_val, y_val))
        conv_mlp_model.fit(X_train, y_train, epochs=30, batch_size=4000, callbacks=[early_stopping], validation_data=(X_val, y_val))
        
        predictions_train_convlstm = conv_lstm_model.predict(X_train)
        predictions_train_convmlp = conv_mlp_model.predict(X_train)
        predictions_test_convlstm = conv_lstm_model.predict(X_test)
        predictions_test_convmlp = conv_mlp_model.predict(X_test)
        predictions_rul_convlstm=conv_lstm_model.predict(last_instances_X)
        predictions_rul_convmlp=conv_mlp_model.predict(last_instances_X)
        
        
        
        #Evaluations
        real_predictions_test_convlstm=predictions_test_convlstm*[(val_max_train[0,13]-val_min_train[0,13])]+val_min_train[0,13]
        real_predictions_test_convmlp=predictions_test_convmlp*[(val_max_train[0,13]-val_min_train[0,13])]+val_min_train[0,13]
        real_y_test=y_test*[(val_max_train[0,13]-val_min_train[0,13])]+val_min_train[0,13]
        real_predictions_rul_convlstm=predictions_rul_convlstm*[(val_max_train[0,13]-val_min_train[0,13])]+val_min_train[0,13]
        real_predictions_rul_convmlp=predictions_rul_convmlp*[(val_max_train[0,13]-val_min_train[0,13])]+val_min_train[0,13]
        rul_y_real=last_instances_y*[(val_max_train[0,13]-val_min_train[0,13])]+val_min_train[0,13]
        
        rmse_test_convlstm=np.sqrt(mean_squared_error(real_predictions_test_convlstm,  (real_y_test)))
        rmse_test_convmlp=np.sqrt(mean_squared_error(real_predictions_test_convmlp,  (real_y_test)))
        rmse_rul_convlstm= np.sqrt(mean_squared_error(real_predictions_rul_convlstm,  (rul_y_real)))
        rmse_rul_convmlp= np.sqrt(mean_squared_error(real_predictions_rul_convmlp,  (rul_y_real)))
        
        mae_test_convlstm=mean_absolute_error(real_predictions_test_convlstm,  (real_y_test))
        mae_test_convmlp=mean_absolute_error(real_predictions_test_convmlp,  (real_y_test))
        mae_rul_convlstm= mean_absolute_error(real_predictions_rul_convlstm,  (rul_y_real))
        mae_rul_convmlp= mean_absolute_error(real_predictions_rul_convmlp,  (rul_y_real))
       
        r2_test_convlstm=r2_score(real_predictions_test_convlstm,  (real_y_test))
        r2_test_convmlp=r2_score(real_predictions_test_convmlp,  (real_y_test))
        r2_rul_convlstm= r2_score(real_predictions_rul_convlstm,  (rul_y_real))
        r2_rul_convmlp= r2_score(real_predictions_rul_convmlp,  (rul_y_real))
        
        
        
        # path = r"C:\Users\rsadeghianbroujeny\.spyder-py3\AnfisTensorflow2.0-master"
        path=r"C:\Users\2411609\OneDrive - Cesi\CESI\Machine virtuelle\AnfisTensorflow2.0-master"
        # path=r"C:\Users\ArRARV\OneDrive - Cesi\CESI\Machine virtuelle\AnfisTensorflow2.0-master"

        os.chdir(path)
        
        from Models import myanfis
        import time
        import Datagenerator.datagenerator as gen
        import datetime
        
     
        

        # import tensorflow.keras.optimizers as optimizers    # <-- for specifying optimizer
        ##############################################################################
        # import math
        X_train_meta=np.concatenate((predictions_train_convlstm, predictions_train_convmlp), axis=1)
        
        X_test_meta=np.concatenate((predictions_test_convlstm,predictions_test_convmlp), axis=1)
        
        # print("GCD of {} and {} is: {}".format(number1, number2, gcd))
        # Model Parameter
        # Model Parameter
        param = myanfis.fis_parameters(
            n_input=2,                # no. of Regressors
            n_memb=50,                 # no. of fuzzy memberships
            batch_size=1,            # 16 / 32 / 64 / ...
            memb_func='gaussian',      # 'gaussian' / 'gbellmf'
            optimizer='adam',          # sgd / adam / ...
            # mse / mae / huber_loss / mean_absolute_percentage_error / ...
            loss='mse',
            n_epochs=10           # 10 / 25 / 50 / 100 / ...
        )
        # Data Parameters
        n_obs = X_train.shape[0]                            # might be adjusted for batch size!
        lag = 10
        data_id = 0                             # 0 = mackey / 1 = sinc /
        # 2 = Three-Input Nonlin /
        # 3 = markov switching
        # 4 = TAR  /  # 5 = STAR
        # General Parameters
        # plt.style.use('seaborn')                # default / ggplot / seaborn
        # plot_prediction = True                  # True / False
        # plot_learningcurves = False              # True / False
        # plot_mfs = True                         # True / False
        # show_initial_weights = True             # True / False
        # plot_heatmap = False                    # True / False
        show_summary = True                     # True / False
        core = '/device:CPU:0'                  # '/device:CPU:0' // '/device:GPU:0'
        show_core_usage = False                 # True / False
        seed = 1                                # set seed for reproducibility
        ##############################################################################
        tf.random.set_seed(seed)
        np.random.seed(seed)
        # Generate Data
        # X, X_train, X_test, y, y_train, y_test = gen.gen_data(
        #     data_id, n_obs, param.n_input, param.batch_size, lag)
        
        # show which devices your operations are assigned to
        tf.debugging.set_log_device_placement(show_core_usage)
        
        with tf.device(core):  # CPU / GPU
            # set tensorboard call back
            log_name = f'-{gen.get_data_name(data_id)}_N{param.n_input}_M{param.n_memb}_batch{param.batch_size}_{param.memb_func}_{param.optimizer}_{param.loss}'
            log_path = os.path.join("logs", "run_anfis",
                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                    + log_name
                                    )
            tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
        
            # create model
            fis = myanfis.ANFIS(n_input=param.n_input,
                                n_memb=param.n_memb,
                                batch_size=param.batch_size,
                                memb_func=param.memb_func,
                                name='myanfis'
                                )
        
            # compile model
            fis.model.compile(optimizer=param.optimizer,
                              loss=param.loss
                              # ,metrics=['mse']  # ['mae', 'mse']
                              )
        
            # fit model
            start_time = time.time()
            history = fis.fit(X_train_meta, y_train,
                              epochs=param.n_epochs,
                              batch_size=param.batch_size,
                              callbacks=[tensorboard_callback]
                              )
           #                      validation_data=(X_test, y_test),
         
            end_time = time.time()
            print(f'Time to fit: {np.round(end_time - start_time,2)} seconds')
        
        # y_preditions_train_meta=fis(X_train_meta)
        # y_prd_testR=fis(X_testR)
        y_preditions_test_meta=fis(X_test_meta)
        
        
        real_predictions_test_meta=y_preditions_test_meta*[(val_max_train[0,13]-val_min_train[0,13])]+val_min_train[0,13]
        rmse_test_meta=np.sqrt(mean_squared_error(real_predictions_test_meta,  (real_y_test)))
        mae_test_meta=mean_absolute_error(real_predictions_test_meta,  (real_y_test))
        r2_test_meta=r2_score(real_predictions_test_meta,  (real_y_test))

        
        print(rmse_test_meta)
        
        error_rmse_convlstm[i-1,k]=rmse_test_convlstm
        error_rmse_convmlp[i-1,k]=rmse_test_convmlp
        error_rmse_meta[i-1,k]=rmse_test_meta

        error_mae_convlstm[i-1,k]=mae_test_convlstm
        error_mae_convmlp[i-1,k]=mae_test_convmlp
        error_mae_meta[i-1,k]=mae_test_meta

        r2_convlstm[i-1,k]=r2_test_convlstm
        r2_convmlp[i-1,k]=r2_test_convmlp
        r2_meta[i-1,k]=r2_test_meta   
        
        # path = r"C:/Users/rsadeghianbroujeny/OneDrive - Cesi/CESI/Machine virtuelle/cmapss"
        path=r"C:\Users\2411609\OneDrive - Cesi\CESI\Machine virtuelle\cmapss\Drive"
        # path=r"C:\Users\ArRARV\OneDrive - Cesi\CESI\Machine virtuelle\cmapss\stacked ensemble learning"

        os.chdir(path)

        filename1=x1 % N
        filename2=x2 % N

        conv_lstm_model.save(filename1)
        conv_mlp_model.save(filename2)

    
np.save('error_rmse_convlstm',error_rmse_convlstm)
np.save('error_rmse_convmlp',error_rmse_convmlp)
np.save('error_rmse_meta',error_rmse_meta)
np.save('error_mae_convlstm',error_mae_convlstm)
np.save('error_mae_convmlp',error_mae_convmlp)
np.save('error_mae_meta',error_mae_meta)
np.save('r2_convlstm',r2_convlstm)
np.save('r2_convmlp',r2_convmlp)
np.save('r2_meta',r2_meta)  

data=[error_rmse[0,:],error_rmse[1,:],error_rmse[2,:],error_rmse[3,:],error_rmse[4,:],error_rmse[5,:],
                      error_rmse[6,:],error_rmse[7,:],error_rmse[8,:],error_rmse[9,:],error_rmse[10,:],error_rmse[11,:],
                      error_rmse[12,:],error_rmse[13,:],error_rmse[14,:],error_rmse[15,:],error_rmse[16,:],error_rmse[17,:],
                            error_rmse[18,:],error_rmse[19,:],error_rmse[20,:],
                            error_rmse[21,:],error_rmse[22,:],error_rmse[23,:],error_rmse[24,:],error_rmse[25,:],error_rmse[26,:],
                                  error_rmse[27,:],error_rmse[28,:],error_rmse[29,:]]
fig, ax = plt.subplots()
meanlineprops = dict(linestyle='--', linewidth=2.5, color='orange')

# Define properties for the median line
medianlineprops = dict(linestyle='-', linewidth=2.5, color='blue')

# Create the boxplot with the mean and median line properties
ax.boxplot(data, showmeans=True, meanline=True, meanprops=meanlineprops, medianprops=medianlineprops)

in1=12427                   
in2=12614
plt.plot(real_y_test[in1:in2],label='Ground truth')

plt.plot(real_predictions_test_meta[in1:in2],label='Prediction_Anfis')

plt.plot(real_predictions_test_convmlp[in1:in2],label='Prediction_ConvMlp')

plt.plot(real_predictions_test_convlstm[in1:in2],label='Prediction_ConvLstm')

plt.legend(loc='lower left', fontsize='large', title_fontsize='medium')


X_test_meta_last_instance=np.concatenate((predictions_rul_convlstm,predictions_rul_convmlp), axis=1)
last_instances_y=last_instances_y*[(val_max_train[0,13]-val_min_train[0,13])]+val_min_train[0,13]

predictions_rul_meta_lst_instances=fis(X_test_meta_last_instance)
predictions_rul_meta_lst_instances_real=predictions_rul_meta_lst_instances*[(val_max_train[0,13]-val_min_train[0,13])]+val_min_train[0,13]
plt.plot(last_instances_y,label='Ground truth')
plt.plot(predictions_rul_meta_lst_instances_real,label='Prediction_Anfis')
plt.plot(real_predictions_rul_convmlp,label='Prediction_ConvMlp')
plt.plot(real_predictions_rul_convlstm,label='Prediction_ConvLstm')
plt.legend(loc='lower left', fontsize='large', title_fontsize='medium')
