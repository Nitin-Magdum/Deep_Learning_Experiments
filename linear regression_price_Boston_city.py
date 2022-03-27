import pandas as pd
import numpy as np
import tensorflow as tf
train_df = pd.read_csv('https://firebasestorage.googleapis.com/v0/b/bible-project-2365c.appspot.com/o/train.csv?alt=media&token=9c5d17c2-0589-43ea-b992-e7c2ad02d714', index_col='ID')
train_df.head()
test_df = pd.read_csv('https://firebasestorage.googleapis.com/v0/b/bible-project-2365c.appspot.com/o/test.csv?alt=media&token=99688b27-9fdb-4ac3-93b8-fa0e0f4d7540', index_col='ID')
test_df.head()
predictors = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
target = 'medv'
from sklearn.preprocessing import MinMaxScaler
print("Note: median values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[13], scaler.min_[13]))
multiplied_by = scaler.scale_[13]
added = scaler.min_[13]
print(type(scaled_train))
scaled_train_df = pd.DataFrame(scaled_train, columns=train_df.columns.values)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
X = scaled_train_df.drop(target, axis=1).values
Y = scaled_train_df[[target]].values# Train the model
model.fit(
    X[10:],
    Y[10:],
    epochs=50,
    shuffle=True,
    verbose=2
)
test_error_rate = model.evaluate(X[:10], Y[:10], verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
prediction = model.predict(X[:1])
y_0 = prediction[0][0]
print('Prediction with scaling - {}',format(y_0))
y_0 -= added
y_0 /= multiplied_by
print("Housing Price Prediction  - ${}".format(y_0))
Y_0 = Y[0]
print('Ground truth with scaling - {}'.format(Y_0))
Y_0 -= added
Y_0 /= multiplied_by
print('Ground Truth Price - ${}'.format(Y_0))
