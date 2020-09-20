import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 4e-4):
            self.model.stop_training = True
callbacks = myCallback()
data = pd.read_csv("total_cases.csv", date_parser=True).fillna(0)
predict_range = 60
data_training = data[data["date"] < "2020-07-18"].copy()
data_testing = data[data["date"] >= "2020-07-18"].copy()

training_data = data_training["Turkey"]
training_data = np.array(training_data)

training_data = training_data.reshape(-1,1)
print(training_data)
scaler = MinMaxScaler()

training_data = scaler.fit_transform(training_data)
x_train = []
y_train = []
for i in range(predict_range, training_data.shape[0]):
    x_train.append(training_data[i-60:i])
    y_train.append(training_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape, y_train.shape)
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(60, activation='relu', return_sequences=True))

model.add(Dropout(0.3))

model.add(LSTM(80, activation='relu'))
model.add(Dense(1))

model.compile(loss="mse", optimizer='adam')
r = model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[callbacks])

past_values = data_training.tail(60)
df = past_values.append(data_testing, ignore_index=True)
df = df["Turkey"]

df = np.array(df).reshape(-1,1)

inputs = scaler.transform(df)
x_test = []
y_test = []
for i in range(predict_range, inputs.shape[0]):
    x_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)
scale_rate = 1/4.61099353e-06
y_pred = y_pred*scale_rate
y_test = y_test*scale_rate
training_data = training_data*scale_rate

data_testing["Predictions"] = y_pred 

forecast = []
new_df = data["Turkey"]
new_df = np.array(new_df)
new_df = new_df.reshape(-1,1)
last_60_days = new_df[-60:]
last_60_days_scaled = scaler.transform(last_60_days)
forecast_test = []
forecast_test.append(last_60_days_scaled)

forecast_test = np.array(forecast_test)

for i in range(60):
    
    pred_price = model.predict(forecast_test)
    forecast_test = np.append(forecast_test, pred_price)

    forecast_test = np.delete(forecast_test, 0)
    
    forecast_test = forecast_test.reshape(1,60,1)
    
    
forecast_test = scale_rate*forecast_test
forecast_test = forecast_test.reshape(60,1)

plt.plot(training_data)
plt.plot(data_testing[['Turkey', 'Predictions']])
k = sum(map(len, [data_testing, training_data]))
plt.plot([*range(k, k+len(forecast_test))], forecast_test, color='red')

plt.title("COVID 19-Predictions")
plt.xlabel("Time")
plt.ylabel("Cases")
plt.legend(["Train", "Real Cases", "AI Predictions", "Forecasting"], loc="lower right")
plt.show()
