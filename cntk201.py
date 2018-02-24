#!/usr/bin/env python3

import matplotlib.pyplot as plt
import math
import numpy as np
import os
import PIL
import sys
from urllib.request import urlopen


import cntk as C
import cntk.io.transforms as xforms

if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

data_path = os.path.join('data', 'CIFAR-10')

# model dimensions
image_height = 32
image_width = 32
num_channels = 3
num_classes = 10

def create_reader(map_file, mean_file, is_training):
    print("Reading map file:", map_file)
    print("Reading mean file:", mean_file)

    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("This tutorials depends 201A tutorials, please run 201A first.")

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    # train uses data augmentation (translation only)
    if is_training:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8)
        ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]
    # deserializer
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        features = C.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = C.io.StreamDef(field='label', shape=num_classes)      # and second as 'label'
    )))


# Create the train and test readers
reader_train = create_reader(os.path.join(data_path, 'train_map.txt'),
                             os.path.join(data_path, 'CIFAR-10_mean.xml'), True)

reader_test = create_reader(os.path.join(data_path, 'test_map.txt'),
                            os.path.join(data_path, 'CIFAR-10_mean.xml'), False)



# Model Creation (Basic CNN)
def create_basic_model(input, out_dims):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        net = C.layers.Convolution((5,5), 32, pad=True)(input)
        net = C.layers.MaxPooling((3,3), strides=(2,2))(net)

        net = C.layers.Convolution((5,5), 32, pad=True)(net)
        net = C.layers.MaxPooling((3,3), strides=(2,2))(net)

        net = C.layers.Convolution((5,5), 64, pad=True)(net)
        net = C.layers.MaxPooling((3,3), strides=(2,2))(net)

        net = C.layers.Dense(64)(net)
        net = C.layers.Dense(out_dims, activation=None)(net)

    return net


def train_and_evaluate(reader_train, reader_test, max_epochs, model_func):
    # Input variables denoting the features and label data
    input_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))

    # Normalize the input
    feature_scale = 1.0 / 256.0
    input_var_norm = C.element_times(feature_scale, input_var)

    # apply model to input
    z = model_func(input_var_norm, out_dims=10)

    #
    # Training action
    #

    # loss and metric
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)

    # training config
    epoch_size     = 50000
    minibatch_size = 64

    # Set training parameters
    lr_per_minibatch       = C.learning_parameter_schedule([0.01]*10 + [0.003]*10 + [0.001],
                                                       epoch_size=epoch_size)
    momentums              = C.momentum_schedule(0.9, minibatch_size=minibatch_size)
    l2_reg_weight          = 0.001

    # trainer object
    learner = C.momentum_sgd(z.parameters,
                             lr = lr_per_minibatch,
                             momentum = momentums,
                             l2_regularization_weight=l2_reg_weight)
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    trainer = C.Trainer(z, (ce, pe), [learner], [progress_printer])

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    C.logging.log_number_of_parameters(z) ; print()

    # perform model training
    batch_index = 0
    plot_data = {'batchindex':[], 'loss':[], 'error':[]}
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count),
                                               input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it

            sample_count += data[label_var].num_samples                     # count samples processed so far

            # For visualization...
            plot_data['batchindex'].append(batch_index)
            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)

            batch_index += 1
        trainer.summarize_training_progress()

    #
    # Evaluation action
    #
    epoch_size     = 10000
    minibatch_size = 16

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)

        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map={
            input_var: reader_test.streams.features,
            label_var: reader_test.streams.labels
        })

        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    # Visualize training result:
    window_width            = 32
    loss_cumsum             = np.cumsum(np.insert(plot_data['loss'], 0, 0))
    error_cumsum            = np.cumsum(np.insert(plot_data['error'], 0, 0))

    # Moving average.
    plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
    plot_data['avg_loss']   = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
    plot_data['avg_error']  = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width

    figure, axes = plt.subplots(2,1)
    axes[0].plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
    axes[0].set_xlabel('Minibatch number')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Minibatch run vs. Training loss ')

    axes[1].plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
    axes[1].set_xlabel('Minibatch number')
    axes[1].set_ylabel('Label Prediction Error')
    axes[1].set_title('Minibatch run vs. Label Prediction Error ')
    plt.tight_layout()
    plt.show()

    return C.softmax(z)


pred = train_and_evaluate(reader_train, reader_test, max_epochs=5, model_func=create_basic_model)



def create_basic_model_terse(input, out_dims):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        model = C.layers.Sequential([
            C.layers.For(range(3), lambda i: [
                C.layers.Convolution((5,5), [32,32,64][i], pad=True),
                C.layers.MaxPooling((3,3), strides=(2,2))
            ]),
            C.layers.Dense(64),
            C.layers.Dense(out_dims, activation=None)
        ])
    return model(input)

pred_basic_model = train_and_evaluate(reader_train, reader_test, max_epochs=10, model_func=create_basic_model_terse)

def evaluate(pred_op, image_data):
    label_lookup =  ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    image_mean = 133.0
    image_data -= image_mean
    image_data = np.ascontiguousarray(np.transpose(image_data, (2,0,1)))
    result = np.squeeze(pred_op.eval({pred_op.arguments[0]:[image_data]}))
    
    top_count = 3
    result_indices = (-np.array(result)).argsort()[:top_count]
    print("Top 3 predictions:")
    for i in range(top_count):
        print("\tLabel: {:10s}, confidence: {:.2f}%".format(label_lookup[result_indices[i]], result[result_indices[i]]* 100))