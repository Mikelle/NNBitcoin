import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


csv = pd.read_csv('btc-real.csv')

training_set = csv.iloc[:, 1:2]
training_set = training_set.values

sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

x_train = training_set[0:2715]
y_train = training_set[1:2716]

today = pd.DataFrame(x_train[0:5])
tomorrow = pd.DataFrame(y_train[0:5])
ex = pd.concat([today,tomorrow], axis=1)
ex.columns = (['today', 'tomorrow'])

x_train = np.reshape(x_train, (2715, 1, 1))

model = Sequential()
model.add(LSTM(units=4, activation='sigmoid', input_shape=(None,1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=2000)

test_set = pd.read_csv('btc-test.csv')

real_stock_price = test_set.iloc[:, 1:2]
real_stock_price = real_stock_price.values

inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (31, 1, 1))
predicted_stock_price = model.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color='blue', label='Real BTC Price')
plt.plot(predicted_stock_price, color='red', label='Predicted BTC Price')
plt.title('BTC Price Prediction')
plt.xlabel('Days')
plt.ylabel('BTC Price')
plt.legend()
plt.show()
