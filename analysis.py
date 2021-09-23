import torch
import numpy as np
from mlp import MLP
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

#from load_data import cascaded_tanks_dataset
from load_data import gas_furnace_dataset
#from load_data import silver_box_dataset
#from load_data import wiener_hammer_dataset

# Load trained model
mlp = MLP(9,1,64)
mlp.load_state_dict(torch.load("trained_models/mlp_gas_furnace_dataset2021-09-22_22-21-04.dat"))

data_name = "gas_furnace_dataset"
x_train, y_train, x_test, y_test = gas_furnace_dataset(4,5,1)
test_inputs = torch.Tensor(x_test)
test_outputs = torch.Tensor(y_test)


preds = []
with torch.no_grad():
    for val in test_inputs:
        y_hat = mlp.forward(val)
        preds.append(y_hat)

preds_arr = np.array(preds)
preds_arr = preds_arr.reshape((test_outputs.shape[0], 1))

# data in original range of values.
max = 60.5 # cascaded_tank: 10 ; gas_furnace: 60.5 ; silverbox :0.26493 ; wiener_hammer: 0.63587
min = 45.6 # cascaded_tank: 2.9116 ; gas_furnace: 45.6 ; silverbox :-0.26249 ; wiener_hammer: -1.1203
scaler = MinMaxScaler(feature_range=(min, max))
preds_arr = scaler.fit_transform(preds_arr)

test_outputs_arr = test_outputs.cpu().detach().numpy()
test_outputs_arr = test_outputs_arr.reshape((test_outputs.shape[0], 1))
test_outputs_arr = scaler.fit_transform(test_outputs_arr)

# MSE
sum = 0
for i in range(test_outputs.shape[0]):
    sum = sum + (preds_arr[i, :] - test_outputs_arr[i, :]) ** 2

mse = sum / test_outputs.shape[0]

print(mse)

plt.plot(test_outputs_arr,label='System',linewidth=1)
plt.plot(preds_arr,'r--', label='MLP',linewidth=1)
plt.title('Output')
plt.xlabel('samples')
plt.ylabel('CO2')
plt.grid()
plt.legend(loc='upper left')
plt.show()
