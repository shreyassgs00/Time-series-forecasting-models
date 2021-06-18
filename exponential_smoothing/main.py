import os
import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing

#Generating dataset
data_file = pd.read_csv('WIT.csv')
data = data_file['Close'].tolist()
index = []
for i in range(0, len(data)):
    index.append(i)
stock_data = pd.Series(data, index)

#Plotting the original dataset
ax=stock_data.plot()
ax.set_xlabel("Samples")
ax.set_ylabel("Stock price")

#Performing exponential smoothing
fit1 = SimpleExpSmoothing(stock_data, initialization_method="heuristic").fit(smoothing_level=0.6, optimized=False)
fcast1 = fit1.forecast(3).rename(r'$\alpha=0.6$')
fit2 = SimpleExpSmoothing(stock_data, initialization_method="estimated").fit()
fcast2 = fit2.forecast(3).rename(r'$\alpha=%s$'%fit2.model.params['smoothing_level'])

#Plotting the results
plt.figure(figsize=(12, 8))
plt.plot(stock_data, marker = 'o', color='black')
plt.plot(fit1.fittedvalues, marker = 'o', color = 'red')
line1, = plt.plot(fcast1, marker = 'o', color = 'red')
plt.plot(fit2.fittedvalues, marker= 'o', color = 'green')
line2, = plt.plot(fcast2, marker='o', color = 'green')
plt.legend([line1, line2], [fcast1.name, fcast2.name])
plt.title('Exponential smoothing plot')
plt.show()

#Error plots and mean squared error
errors = []
for i in range(0, len(data)):
    errors.append(fit2.fittedvalues[i] - data[i])
plt.figure()
plt.plot(errors)
plt.xlabel('Time')
plt.ylabel('Errors')
plt.title('Error plot')
plt.show()

#Finding mean squared error values
mse = 0
for j in range(0, len(data)):
    mse = mse + math.sqrt(abs((fit2.fittedvalues[i])**2 - data[i]**2))
print('Mean squared error is: ', mse)
