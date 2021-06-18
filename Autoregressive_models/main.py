import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf

data = pd.read_csv('month_dataset/RS.csv')
plt.plot(data['Close'])
plt.xlabel('Samples')
plt.ylabel('Stock price (in $)')
plt.title('Actual data')
plt.show()

differenced_data = []
raw_data = data['Close'].tolist()
for i in range(0, len(raw_data)-1):
    differenced_data.append(raw_data[i+1]-raw_data[i])

data_stationarity = adfuller(differenced_data, autolag = 'AIC')
print('ADF Statistic: ', data_stationarity[0])
print('P-value: ', data_stationarity[1])
for key,value in data_stationarity[4].items():
    print('Critical value: ')
    print(f'    {key}, {value}')

pacf = plot_pacf(differenced_data, lags=9)
plt.title('Partial autocorrelation plot')
plt.show()


train_data = data['Close'][:len(data)-5]
test_data = data['Close'][len(data)-5:]

ar_model = AutoReg(train_data, lags = 2).fit()

predictions = ar_model.predict( start=len(train_data), end=(len(data)-1), dynamic=False)
print(predictions)
plt.plot(predictions, color = 'yellow')
plt.plot(test_data, color = 'black')
plt.title('Actual vs predicted data')
plt.xlabel('Samples')
plt.ylabel('Stock price (in $)')
plt.legend(['Predicted','Actual'])
plt.show()

test_data = data['Close'][len(data)-5:].tolist()

mean_squared_error = 0
for k in range(0, len(test_data)):
    mean_squared_error += math.sqrt(abs(predictions[k+17]**2 - test_data[k]**2))
print(mean_squared_error)

error = []
for l in range(0, len(test_data)):
    error.append(predictions[l+17]-test_data[l])
plt.plot(error)
plt.title('Error plot')
plt.xlabel('Samples')
plt.ylabel('Error')
plt.show()

percentage_accuracy = 0