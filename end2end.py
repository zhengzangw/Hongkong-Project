import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import svm
from tensorflow.keras import regularizers

os.environ['KMP_DUPLICATE_LIB_OK']='True'
MAKECLS = ['n_samples','n_features','n_classes','n_clusters_per_class','n_informative','flip_y','scale']
CLS = ['alpha','max_iter','n_jobs','l1','l2','alpha_inv']

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,3])
  plt.show()

def plot_y(y_test, y_pred):
  plt.scatter(y_test, y_pred)
  plt.xlabel('True Values [MPG]')
  plt.ylabel('Predictions [MPG]')
  plt.axis('equal')
  plt.axis('square')
  plt.xlim([0,plt.xlim()[1]])
  plt.ylim([0,plt.ylim()[1]])
  plt.plot([-100, 100], [-100, 100])
  plt.show()

def generate_model():
  model = keras.Sequential([
    layers.BatchNormalization(input_shape=[len(df.keys())-1]),
    layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam()

  model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
  model.summary()
  return model

def regressor(df):
  y = df[["time"]]
  X = df.drop(["time"], axis=1)

  x_stats = X.describe().transpose()
  # X = norm(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)
  y_stats = y_train.describe().transpose()
  # y_train = norm(y_train)

  model = generate_model()
  early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
  history = model.fit(X_train, y_train,
    epochs=EPOCHS, validation_split = 0.15, verbose=0, callbacks=[early_stop])
  plot_history(history)

  y_pred_train = model.predict(X_train)
  y_pred_train[y_pred_train<0] = y_stats['min']
  mse_train = mean_squared_error(y_train, y_pred_train)
  print("MSE of train is {}".format(mse_train))

  y_pred = model.predict(X_test)
  # y_pred = rev_norm(y_pred, y_stats)
  y_pred[y_pred<0] = y_stats['min']
  mse = mean_squared_error(y_test, y_pred)
  print("MSE of test is {}".format(mse))

  plot_y(y_test, y_pred)

  return model, x_stats, y_stats

np.random.seed(1234)
EPOCHS = 100000

if __name__=="__main__":
  
  df = pd.read_csv("train_modified.csv")

  reg, x_stat, y_stat = regressor(df)

  df_test = pd.read_csv('test_modified.csv')
  sub = pd.read_csv('sample_submission.csv')
  sub['time'] = reg.predict(df_test)
  sub.loc[sub['time']<0,'time'] = y_stat['min'][0]
  sub.to_csv('submission.csv', index=False)