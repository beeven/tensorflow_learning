import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cntk as C

isFast = True

def create_reader(path, is_training, input_dim, num_label_classes):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        labels = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
        features = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

train_file = "./MNIST-data/Train-28x28_cntk_text.txt"
test_file = "./MNIST-data/Test-28x28_cntk_text.txt"

if not os.path.isfile(train_file) or not os.path.isfile(test_file):
    raise ValueError("Please generate the data first")




input_dim = 784
encoding_dims = [128, 64, 32]
decoding_dims = [64, 128]

encoded_model = None

def create_deep_model(features):
    with C.layers.default_options(init=C.layers.glorot_uniform()):
        encode = C.element_times(C.constant(1.0/255.0), features)
        for encoding_dim in encoding_dims:
            encode = C.layers.Dense(encoding_dim, activation=C.relu)(encode)
        global encoded_model
        encoded_model = encode
        decode = encode
        for decoding_dim in decoding_dims:
            decode = C.layers.Dense(decoding_dim, activation=C.relu)(decode)
        decode = C.layers.Dense(input_dim, activation=C.sigmoid)(decode)
        return decode


def train_and_test(reader_train, reader_test, model_func):
    
    ###############################
    # Training the model
    ###############################

    input = C.input_variable(input_dim)
    label = C.input_variable(input_dim)

    model = model_func(input)

    target = label/255.0
    loss = -(target * C.log(model) + (1 - target)*C.log(1-model))
    label_error = C.classification_error(model, target)

    epoch_size = 30000
    minibatch_size = 64
    num_sweeps_to_train_with = 5 if isFast else 100
    num_samples_per_sweep = 60000
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) // minibatch_size

    lr_per_sample = [3e-4]
    lr_schedule = C.learning_parameter_schedule_per_sample(lr_per_sample, epoch_size)

    momentum_schedule = C.momentum_schedule(0.9126265014311797, minibatch_size)

    learner = C.fsadagrad(model.parameters, lr=lr_schedule, momentum=momentum_schedule)

    progress_printer = C.logging.ProgressPrinter(0)
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    input_map = {
        input: reader_train.streams.features,
        label: reader_train.streams.features
    }

    aggregate_metric = 0
    for i in range(num_minibatches_to_train):
        data = reader_train.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(data)
        samples = trainer.previous_minibatch_sample_count
        aggregate_metric += trainer.previous_minibatch_evaluation_average * samples
    
    train_error = (aggregate_metric * 100) / (trainer.total_number_of_samples_seen)
    print("Average training error: {0:0.2f}%".format(train_error))

    #############################################################################
    # Testing the model
    # Note: we use a test file reader to read data different from a training data
    #############################################################################

    test_minibatch_size = 32
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    test_result = 0

    # Test error metric calculation
    metric_numer = 0
    metric_denom = 0

    test_input_map = {
        input: reader_test.streams.features,
        label: reader_test.streams.features
    }

    for i in range(0, int(num_minibatches_to_test)):
        data = reader_test.next_minibatch(test_minibatch_size, input_map=test_input_map)
        eval_error = trainer.test_minibatch(data)
        metric_numer += np.abs(eval_error * test_minibatch_size)
        metric_denom += test_minibatch_size
    test_error = (metric_numer * 100) / (metric_denom)
    print("Average test error: {0:0.2f}%".format(test_error))

    return model, train_error, test_error 

num_label_classes = 10
reader_train = create_reader(train_file, True, input_dim, num_label_classes)
reader_test = create_reader(test_file, False, input_dim, num_label_classes)
reader_eval = create_reader(test_file, False, input_dim, num_label_classes)

model, deep_ae_train_error, deep_ae_test_error = train_and_test(reader_train, reader_test, model_func=create_deep_model)


# Visualize deep AE results

eval_minibatch_size = 50
eval_input_map = { input: reader_eval.streams.features }

eval_data = reader_eval.next_minibatch(eval_minibatch_size, input_map=eval_input_map)

img_data = eval_data[input].asarray()
idx = np.random.choice(eval_minibatch_size)

orig_image = img_data[idx,:,:]
decoded_image = model.eval(orig_image)[0]*255

def print_image_stats(img, text):
    print(text)
    print("Max : {0:.2f}, Median: {1:.2f}, Mean: {2:.2f}, Min: {3:.2f}".format(np.max(img),
                                                                               np.median(img),
                                                                               np.mean(img),
                                                                               np.min(img)))

def plot_image_pair(img1, text1, img2, text2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,6))
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title(text1)
    axes[0].axis("off")
    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title(text2)
    axes[1].axis("off")
    plt.show()

print_image_stats(orig_image, "Original image statistics:")
print_image_stats(decoded_image, "Decoded image statistics:")

img1 = orig_image.reshape(28,28)
text1 = 'Original image'

img2 = decoded_image.reshape(28,28)
text2 = 'Decoded image'

# plot_image_pair(img1, text1, img2, text2)



reader_viz = create_reader(test_file, False, input_dim, num_label_classes)
image = C.input_variable(input_dim)
image_label = C.input_variable(num_label_classes)

viz_minibatch_size = 50
viz_input_map = {
    image: reader_viz.streams.features,
    image_label: reader_viz.streams.labels
}

viz_data = reader_viz.next_minibatch(viz_minibatch_size, input_map = viz_input_map)

img_data = viz_data[image].asarray()
imglabel_raw = viz_data[image_label].asarray()

img_labels = [np.argmax(imglabel_raw[i,:,:]) for i in range(0, imglabel_raw.shape[0])]

from collections import defaultdict
label_dict = defaultdict(list)
for img_idx, img_label in enumerate(img_labels):
    label_dict[img_label].append(img_idx)

randIdx = [1,3,9]
for i in randIdx:
    print("{0}: {1}".format(i, label_dict[i]))


from scipy import spatial

def image_pair_cosine_distance(img1, img2):
    if img1.size != img2.size:
        raise ValueError("Two images need to be of same dimension")
    return 1 - spatial.distance.cosine(img1, img2)

digist_of_interest = np.random.randint(0,9)
digit_index_list = label_dict[digist_of_interest]

if len(digit_index_list) < 2:
    print("Need at least two images to compare")
else:
    imgA = img_data[digit_index_list[0],:,:][0]
    imgB = img_data[digit_index_list[1],:,:][0]
    imgA_B_dist = image_pair_cosine_distance(imgA, imgB)
    print("Distance between two original image: {0:.3f}".format(imgA_B_dist))

    # plot_image_pair(img1, text1, img2, text2)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
    axes[0,0].imshow(imgA.reshape(28,28))
    axes[0,0].set_title('Original image 1')
    axes[0,0].axis("off")
    axes[0,1].imshow(imgB.reshape(28,28))
    axes[0,1].set_title('Original image 2')
    axes[0,1].axis("off")


    imgA_decoded = model.eval([imgA])[0]
    imgB_decoded = model.eval([imgB])[0]
    imgA_B_decoded_dist = image_pair_cosine_distance(imgA_decoded, imgB_decoded)

    print("Distance between two decoded image: {0:.3f}".format(imgA_B_decoded_dist))


    axes[1,0].imshow(imgA_decoded.reshape(28,28))
    axes[1,0].set_title('Decoded image 1')
    axes[1,0].axis("off")
    axes[1,1].imshow(imgB_decoded.reshape(28,28))
    axes[1,1].set_title('Decoded image 2')
    axes[1,1].axis("off")

    plt.show()