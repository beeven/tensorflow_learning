import cntk as C
import numpy as np
from urllib.request import urlretrieve
import gzip
import os
import sys
import struct

def loadData(gzfname):
    with gzip.open(gzfname) as gz:
        n = struct.unpack('I', gz.read(4))
        if n[0] != 0x3080000:
            raise Exception('Invalid file: unexpected magic number.')
        cimg = struct.unpack('>I', gz.read(4))[0]
        
        crow = struct.unpack('>I', gz.read(4))[0]
        ccol = struct.unpack('>I', gz.read(4))[0]
        if crow !=28 or ccol != 28:
            raise Exception('Invalid file: expected 28 rows/cols per image.')
        res = np.frombuffer(gz.read(cimg * crow * ccol), dtype = np.uint8)
        return res.reshape((cimg, crow * ccol))

def loadLabels(gzfname):
    with gzip.open(gzfname) as gz:
        n = struct.unpack('I', gz.read(4))
        if n[0] != 0x1080000:
            raise Exception('Invalid file: unexpected magic number.')
        cimg = struct.unpack('>I', gz.read(4))[0]
        res = np.frombuffer(gz.read(cimg), dtype=np.uint8)
        return res.reshape((cimg, 1))

def savetxt(filename, ndarray):
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    print("Saving", filename)
    with open(filename, "w") as f:
        labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
        for row in ndarray:
            row_str = row.astype(str)
            label_str = labels[row[-1]]
            feature_str = ' '.join(row_str[:-1])
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))

def create_reader(path, is_training, input_dim, num_label_classes):
    ctf = C.io.CTFDeserializer(path, C.io.StreamDefs(
        labels=C.io.StreamDef(field='labels',shape=num_label_classes, is_sparse=False),
        features=C.io.StreamDef(field='features',shape=input_dim, is_sparse=False)))
    return C.io.MinibatchSource(ctf, randomize=is_training, max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1)

def prepare_data():
    dirname = "./MNIST-data"
    os.makedirs(dirname, exist_ok=True)
    base_addr = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    for fname in files:
        if not os.path.exists(os.path.join(dirname,fname)):
            print("Downloading "+base_addr+fname)
            urlretrieve(base_addr+fname, os.path.join(dirname,fname))
    train_data = loadData(os.path.join(dirname,files[0]))
    train_labels = loadLabels(os.path.join(dirname,files[1]))
    test_data = loadData(os.path.join(dirname,files[0]))
    test_labels = loadLabels(os.path.join(dirname,files[1]))
    train_file = os.path.join(dirname, "Train-28x28_cntk_text.txt")
    test_file = os.path.join(dirname, "Test-28x28_cntk_text.txt")
    if not os.path.exists(train_file):
        savetxt(train_file, np.hstack(train_data, train_labels))
    if not os.path.exists(test_file):
        savetxt(test_file, np.hstack(test_data, test_labels))
    train_reader = create_reader(train_file, True, 784, 10)
    test_reader = create_reader(test_file, False, 784, 10) 
    return train_reader, test_reader   
    

def model_func(features):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.ops.relu):
        h = features
        h = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=32,
                                   strides=(1,1),
                                   pad=True,
                                   name='first_conv')(h)
        h = C.layers.MaxPooling(filter_shape=(2,2),
                                strides=(2,2),
                                name='first_max')(h)
        h = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=64,
                                   strides=(1,1),
                                   pad=True,
                                   name='second_conv')(h)
        h = C.layers.MaxPooling(filter_shape=(2,2),
                                strides=(2,2),
                                name='second_max')(h)
        h = C.layers.Dense(1024)(h)
        h = C.layers.Dropout(0.4)(h)
        r = C.layers.Dense(10)(h)
        return r
                                
def main():
    train_reader, test_reader = prepare_data()
    x = C.input_variable((1,28,28))
    y = C.input_variable(10)
    z = model_func(x)
    model = z(x/255)
    loss = C.cross_entropy_with_softmax(model, y)
    label_error = C.classification_error(model, y)
    learning_rate = 0.2
    lr_schedule = C.learning_parameter_schedule(learning_rate)
    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, label_error), [learner])

    minibatch_size = 64
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 10
    num_minibatches_to_train = 2000 #(num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    for i in range(0, int(num_minibatches_to_train)):
        data = train_reader.next_minibatch(minibatch_size, {
            x: train_reader.streams.features,
            y: train_reader.streams.labels
        })
        trainer.train_minibatch(data)
        if i % 100 == 0:
            training_loss = trainer.previous_minibatch_loss_average
            eval_error = trainer.previous_minibatch_evaluation_average
            print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(
                i, training_loss, eval_error*100
            ))
    
    # Evaluate the model
    test_minibatch_size=512
    num_samples = 10000
    num_minibatches_to_test = num_samples // test_minibatch_size
    test_result = 0.0
    for i in range(num_minibatches_to_test):
        data = test_reader.next_minibatch(test_minibatch_size, input_map={
            x: test_reader.streams.features,
            y: test_reader.streams.labels
        })
        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error
    print("Average test error: {0:.2f}%".format(test_result*100/num_minibatches_to_test))


if __name__=="__main__":
    main()