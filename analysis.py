import torch
import numpy as np
from mlp import MLP
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from load_data import cascaded_tanks_dataset
from load_data import gas_furnace_dataset
from load_data import silver_box_dataset
from load_data import wiener_hammer_dataset

# Load trained model
mlp = MLP(9,1,16)
mlp.load_state_dict(torch.load("trained_models/mlp2021-09-09_00-53-23.dat"))

x_train, y_train, x_test, y_test = cascaded_tanks_dataset(4,5,1)
test_inputs = torch.Tensor(x_test)
test_outputs = torch.Tensor(y_test)


preds = []
with torch.no_grad():
    for val in test_inputs:
        y_hat = mlp.forward(val)
        preds.append(y_hat)

preds_arr = np.array(preds)
preds_arr = preds_arr.reshape((1019, 1))

test_outputs_arr = test_outputs.cpu().detach().numpy()
test_outputs_arr = test_outputs_arr.reshape((1019, 1))
# MSE
sum = 0
for i in range(1019):
    sum = sum + (preds_arr[i, :] - test_outputs_arr[i, :]) ** 2

mse = sum / 1019

print(mse)

plt.plot(test_outputs,label='validation data',linewidth=1)
plt.plot(preds,'r--', label='mlp reconstruction',markersize=1)
plt.title('MLP validation')
plt.xlabel('samples')
plt.ylabel('magnitude (V)')
plt.grid()
plt.legend()
plt.show()