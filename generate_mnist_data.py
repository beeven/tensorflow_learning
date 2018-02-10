import gzip
import struct
import numpy as np
import os
from urllib.request import urlretrieve

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
        labels = list(map(' '.join, np.eye(10, dtype=np.uint8).astype(str)))
        for row in ndarray:
            row_str = row.astype(str)
            label_str = labels[row[-1]]
            feature_str = ' '.join(row_str[:-1])
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))

if __name__=="__main__":
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
    print("Done")
