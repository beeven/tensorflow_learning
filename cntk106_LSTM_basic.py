import math
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import cntk as C

isFast = True

# Data generation

def split_data(data, val_size=0.1, test_size=0.1):
    """
    split np.array into training, validation and test
    """
    pos_test = int(len(data) * (1-test_size))
    pos_val = int(len(data[:pos_test]) * (1 - val_size))

    train, val, test = data[:pos_val], data[pos_val:pos_test], data[pos_test:]

    return { "train": train, "val": val, "test": test }

def generate_data(fct, x, time_steps, time_shift):
    """
    generate sequences to feed to rnn for fct(x)
    """
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(dict(a = data[0: len(data) - time_shift],
                                 b = data[time_shift:]))
    rnn_x = []
    for i in range(len(data) - time_steps + 1):
        rnn_x.append(data['a'].iloc[i: i + time_steps].as_matrix())
    rnn_x = np.array(rnn_x)

    # Reshape or rearrange the data from row to column
    # to be compatible with the input needed by the LSTM model
    # which expects 1 float per time point in a given batch
    rnn_x = rnn_x.reshape(rnn_x.shape + (1,))

    rnn_y = data['b'].values
    rnn_y = rnn_y[time_steps - 1:]
    
    # Reshape or rearrange the data from row to columns
    # to match the input shape
    rnn_y = rnn_y.reshape(rnn_y.shape + (1,))
    
    return split_data(rnn_x), split_data(rnn_y)

N = 5 # input: N subsequent values
M = 5 # output: predict 1 value M steps ahead
X, Y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), N, M)

def view_sample():
    f, a = plt.subplots(3, 1, figsize=(12,8))
    for j, ds in enumerate(["train", "val", "test"]):
        a[j].plot(Y[ds], label=ds + 'raw')
    [i.legend() for i in a];
    plt.show()

#view_sample()


# Network modeling

def create_model(x):
    """ Create the model for time series prediction """
    with C.layers.default_options(initial_state=0.1):
        m = C.layers.Recurrence(C.layers.LSTM(N))(x)
        m = C.sequence.last(m)
        m = C.layers.Dropout(0.2, seed=1)(m)
        m = C.layers.Dense(1)(m)
        return m


# Train the network

TRAINING_STEPS = 10000
BATCH_SIZE = 100
EPOCHS = 10 if isFast else 100

def next_batch(x, y, ds):
    """ Get the next batch to process """
    def as_batch(data, start, count):
        part = []
        for i in range(start, start + count):
            part.append(data[i])
        return np.array(part)

    for i in range(0, len(x[ds]) - BATCH_SIZE, BATCH_SIZE):
        yield as_batch(x[ds], i, BATCH_SIZE), as_batch(y[ds], i, BATCH_SIZE)


## Key Insight

# input sequences
x = C.sequence.input_variable(1)
# create the model
z = create_model(x)

# expected output (label), also the dynamic axes of the model output
# is specified as the model of the label input
l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")

learning_rate = 0.02
lr_schedule = C.learning_parameter_schedule(learning_rate)

loss = C.squared_error(z, l)
error = C.squared_error(z, l)

momentum_schedule = C.momentum_schedule(0.9, minibatch_size=BATCH_SIZE)
learner = C.fsadagrad(z.parameters, lr=lr_schedule, momentum=momentum_schedule, unit_gain=True)

trainer = C.Trainer(z, (loss, error), [learner])

loss_summary = []
start = time.time()
for epoch in range(0, EPOCHS):
    for x1, y1 in next_batch(X, Y, "train"):
        trainer.train_minibatch({x: x1, l: y1})
    if epoch % (EPOCHS / 10) == 0:
        training_loss = trainer.previous_minibatch_loss_average
        loss_summary.append(training_loss)
        print("epoch: {}, loss: {:.5f}".format(epoch, training_loss))

print("training took {0:.1f} sec".format(time.time() - start))

def view_train_loss():
    plt.plot(loss_summary, label='training loss')
    plt.show()

#view_train_loss()

def get_mse(X,Y, labeltxt):
    result = 0.0
    for x1, y1 in next_batch(X, Y, labeltxt):
        eval_error = trainer.test_minibatch({x: x1, l: y1})
        result += eval_error
    return result / len(X[labeltxt])

for labeltxt in ["train", "val", "test"]:
    print("mse for {}: {:.6f}".format(labeltxt, get_mse(X, Y, labeltxt)))


f, a = plt.subplots(3,1,figsize=(12,8))
for j, ds in enumerate(["train", "val", "test"]):
    results = []
    for x1, y1 in next_batch(X, Y, ds):
        pred = z.eval({x: x1})
        results.extend(pred[:, 0])
    a[j].plot(Y[ds], label = ds + ' raw')
    a[j].plot(results, label = ds + ' predicted')
[i.legend() for i in a];
plt.show()