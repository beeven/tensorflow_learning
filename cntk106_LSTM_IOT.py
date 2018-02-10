#!/usr/bin/env python3

from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time
import cntk as C
from urllib.request import urlretrieve

isFast = False

EPOCHS = 100 if isFast else 2000


def generate_solar_data(input_url, time_steps, normalize=1, val_size=0.1, test_size=0.1):
    """
    generate sequences to feed to rnn based on data frame with solar panel data
    the csv has the format: time ,solar.current, solar.total
     (solar.current is the current output in Watt, solar.total is the total production
      for the day so far in Watt hours)
    """
    cache_path = os.path.join("data", "iot")
    cache_file = os.path.join(cache_path, "solar.csv")
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(cache_file):
        print("downloading data from ", input_url)
        urlretrieve(input_url, cache_file)
        print("downloaded data successfully")
    else:
        print("using cache for ", input_url)

    df = pd.read_csv(cache_file, index_col="time",
                     parse_dates=['time'], dtype=np.float32)
    df["date"] = df.index.date

    df['solar.current'] /= normalize
    df['solar.total'] /= normalize

    grouped = df.groupby(df.index.date).max()
    grouped.columns = ["solar.current.max", "solar.total.max", "date"]

    df_merged = pd.merge(df, grouped, right_index=True, on="date")
    df_merged = df_merged[["solar.current", "solar.total",
                           "solar.current.max", "solar.total.max"]]

    grouped = df_merged.groupby(df_merged.index.date)
    per_day = []
    for _, group in grouped:
        per_day.append(group)

    val_size = int(len(per_day) * val_size)
    test_size = int(len(per_day) * test_size)
    next_val = 0
    next_test = 0

    result_x = {"train": [], "val": [], "test": []}
    result_y = {"train": [], "val": [], "test": []}

    for i, day in enumerate(per_day):
        total = day["solar.total"].values
        if len(total) < 8:
            continue
        if i >= next_val:
            current_set = "val"
            next_val = i + int(len(per_day) / val_size)
        elif i >= next_test:
            current_set = "test"
            next_test = i + int(len(per_day) / test_size)
        else:
            current_set = "train"
        max_total_for_day = np.array(day["solar.total.max"].values[0])
        for j in range(2, len(total)):
            result_x[current_set].append(total[0:j])
            result_y[current_set].append([max_total_for_day])
            if j >= time_steps:
                break

    for ds in ["train", "val", "test"]:
        result_y[ds] = np.array(result_y[ds])
    return result_x, result_y


TIMESTEPS = 14
NORMALIZE = 20000
X, Y = generate_solar_data(
    "https://www.cntk.ai/jup/dat/solar.csv", TIMESTEPS, normalize=NORMALIZE)

BATCH_SIZE = TIMESTEPS * 10

def next_batch(x, y, ds):
    """ get the next batch of training """
    def as_batch(data, start, count):
        return data[start: start+count]
    for i in range(0, len(x[ds]), BATCH_SIZE):
        yield as_batch(x[ds], i, BATCH_SIZE), as_batch(y[ds], i, BATCH_SIZE) 

H_DIMS = 15

def create_model(x):
    """ create the model for time series prediction """
    with C.layers.default_options(initial_state = 0.1):
        m = C.layers.Recurrence(C.layers.LSTM(H_DIMS))(x)
        m = C.sequence.last(m)
        m = C.layers.Dropout(0.2)(m)
        m = C.layers.Dense(1)(m)
        return m

# Training

x = C.sequence.input_variable(1)
z = create_model(x)
l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")

learning_rate = 5e-3
lr_schedule = C.learning_parameter_schedule(learning_rate)
loss = C.squared_error(z, l)
error = C.squared_error(z, l)
momentum_schedule = C.momentum_schedule(0.9, minibatch_size=BATCH_SIZE)
learner = C.fsadagrad(z.parameters, lr=lr_schedule, momentum=momentum_schedule)
trainer = C.Trainer(z, (loss, error), [learner])

C.logging.log_number_of_parameters(z)

loss_summary = []
start = time.time()
for epoch in range(0, EPOCHS):
    for x_batch, l_batch in next_batch(X,Y, "train"):
        trainer.train_minibatch({x: x_batch, l: l_batch})
    
    #if epoch % (EPOCHS /10) == 0:
    training_loss = trainer.previous_minibatch_loss_average
    loss_summary.append(training_loss)
    print("epoch: {}, loss: {:.4f}".format(epoch, training_loss))
print("Training took {:.1f} sec".format(time.time()-start))

# plt.plot(loss_summary, label='training loss')
# plt.show()

def get_mse(X, Y, labeltxt):
    result = 0.0
    for x1, y1 in next_batch(X, Y, labeltxt):
        eval_error = trainer.test_minibatch({x: x1, l: y1})
        result += eval_error
    return result/len(X[labeltxt])

for labeltxt in ["train", "val", "test"]:
    print("mse for {}: {:.6f}".format(labeltxt, get_mse(X, Y, labeltxt)))


# Visualize results

f, a = plt.subplots(2, 1, figsize=(12,8))
for j, ds in enumerate(["val", "test"]):
    results = []
    for x_batch, _ in next_batch(X,Y,ds):
        pred = z.eval({x: x_batch})
        results.extend(pred[:, 0])
    
    a[j].plot((Y[ds] * NORMALIZE).flatten(), label=ds+' raw')
    a[j].plot(np.array(results) * NORMALIZE, label=ds+' pred')
    a[j].legend()

plt.show()