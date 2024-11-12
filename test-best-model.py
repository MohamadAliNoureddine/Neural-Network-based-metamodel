import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

"""
best_save_path = "./Site-Station/"
best_model_path = f"{best_save_path}/best_model_epoch_197.h5"
best_model = tf.keras.models.load_model(best_model_path)
"""
best_save_path = "./Reference-Station"
best_model_path = f"{best_save_path}/best_model_epoch_179.h5"
best_model = tf.keras.models.load_model(best_model_path)

Uncertain_validation_inputs_path_file= "./Reference-Station/validation-set/Inputs-validation-simulations.txt"
Uncertain_validation_inputs= np.genfromtxt(Uncertain_validation_inputs_path_file)
x_test=Uncertain_validation_inputs
 # Faire des prédictions
y_pred_best=best_model.predict(x_test)

def read_efi_binary_time_series_ux(myfiles):
    dt = np.dtype([('temps', 'f4'), ('ux', 'f4'), ('uy', 'f4'), ('uz', 'f4'), ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'), ('ax', 'f4'), ('ay', 'f4'), ('az', 'f4')])
    with open(myfiles, "rb") as f:
        x = np.fromfile(f, dtype=dt)
        temps = x['temps'].astype('float32')
        ux = x['ux'].astype('float32')
    return temps, ux
directory_path = "./Reference-Station/train-set"
num_simulations = 500

simulation_data1 = []
for i in range(1, num_simulations + 1):
    simu_name = f"SIMU{str(i).zfill(6)}"
    file_path = f"{directory_path}/{simu_name}/dumanoirwithdam.fsr.000004.gpl.filter"
    temps, ux = read_efi_binary_time_series_ux(file_path)
    simulation_data1.append((temps, ux))

num_components = 2

num_temps_points =1001
tensor_data_ux = np.zeros((num_simulations, num_components, num_temps_points))
for i, data1 in enumerate(simulation_data1):
    temps = data1[0]
    ux = data1[1]
    tensor_data_ux[i, 0, :] = temps
    tensor_data_ux[i, 1, :] = ux
temps = tensor_data_ux[:, 0, :]
temps= np.squeeze(temps)
displacement_ux = tensor_data_ux[:, 1, :]
displacement_ux = np.squeeze(displacement_ux)
y_train_ux = displacement_ux
y_min_ux = np.min(y_train_ux)
y_max_ux = np.max(y_train_ux)
y_pred_best=y_pred_best* (y_max_ux - y_min_ux) + y_min_ux
directory_test_path = "./Reference-Station/validation-set"
num_simulations = 100

simulation_data1 = []
for i in range(501, num_simulations + 501):
    simu_name = f"SIMU{str(i).zfill(6)}"
    file_path = f"{directory_test_path}/{simu_name}/dumanoirwithdam.fsr.000004.gpl.filter"
    temps, ux = read_efi_binary_time_series_ux(file_path)
    simulation_data1.append((temps, ux))

num_components = 2

num_temps_points =1001
tensor_data_ux = np.zeros((num_simulations, num_components, num_temps_points))
for i, data1 in enumerate(simulation_data1):
    temps = data1[0]
    ux = data1[1]
    tensor_data_ux[i, 0, :] = temps
    tensor_data_ux[i, 1, :] = ux
temps = tensor_data_ux[:, 0, :]
temps= np.squeeze(temps)
velocities_ux = tensor_data_ux[:, 1, :]
velocities_ux = np.squeeze(velocities_ux)
y_test=velocities_ux
for i in range(100):
    plt.figure()
    plt.plot(temps[0 ], y_pred_best[i],linewidth=2, markersize=6, label=f'best-NN')
    plt.plot(temps[0], y_test[i],linewidth=2, markersize=6,label='SEM')
    plt.xlabel('time(s)', fontsize=14, fontname='Arial')
    plt.ylabel('EW-displacement[m/s/Hz]', fontsize=14, fontname='Arial')
    plt.legend(loc='upper right', fontsize=12)
    plt.show()

