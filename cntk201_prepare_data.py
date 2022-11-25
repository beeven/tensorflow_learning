
# Use a function definition from future version (say 3.x from 2.7 interpreter)
from __future__ import print_function

from PIL import Image
import getopt
import numpy as np
import pickle as cp
import os
import shutil
import struct
import sys
import tarfile
import xml.etree.cElementTree as et
import xml.dom.minidom

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


imgSize = 32
numFeatures = imgSize * imgSize * 3


def readBatch(src):
    with open(src, 'rb') as f:
        d = cp.load(f, encoding='latin1')
        data = d['data']
        feat = data
    res = np.hstack((feat, np.reshape(d['labels'], (len(d['labels']), 1))))
    return res.astype(np.int)


def loadData(src, dest):
    if os.path.exists(dest):
        print("Data file exists, reuse cached data file.")
        fname = dest
    else:
        print('Downloading ' + src)
        fname, _ = urlretrieve(src, dest)
        print('Done.')

    root_dir = os.path.dirname(dest)

    print('Extracting files...')
    with tarfile.open(fname) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=root_dir)
    print('Done.')
    print('Preparing train set...')
    trn = np.empty((0, numFeatures + 1), dtype=np.int)
    for i in range(5):
        batchName = os.path.join(
            root_dir, 'cifar-10-batches-py', 'data_batch_{0}'.format(i+1))
        trn = np.vstack((trn, readBatch(batchName)))
    print('Done.')
    print('Preparing test set...')
    tst = readBatch(os.path.join(
        root_dir, 'cifar-10-batches-py', 'test_batch'))
    return (trn, tst)


def saveTxt(filename, ndarray):
    with open(filename, 'w') as f:
        labels = list(map(' '.join, np.eye(10, dtype=np.uint8).astype(str)))
        for row in ndarray:
            row_str = row.astype(str)
            label_str = labels[row[-1]]
            feature_str = ' '.join(row_str[:-1])
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))


def saveImage(fname, data, label, mapFile, regrFile, pad, **key_parms):
    # data in CIFAR-10 dataset is in CHW format.
    pixData = data.reshape((3, imgSize, imgSize))
    if 'mean' in key_parms:
        key_parms['mean'] += pixData

    if pad > 0:
        pixData = np.pad(pixData, ((0, 0), (pad, pad), (pad, pad)),
                         mode='constant', constant_values=128)

    img = Image.new('RGB', (imgSize + 2 * pad, imgSize + 2 * pad))
    pixels = img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            pixels[x, y] = (pixData[0][y][x], pixData[1][y][x], pixData[2][y][x])
    img.save(fname)
    mapFile.write("%s\t%d\n" % (fname, label))

    # compute per channel mean and store for regression example
    channelMean = np.mean(pixData, axis=(1,2))
    regrFile.write("|regrLabels\t%f\t%f\t%f\n" % (channelMean[0]/255.0, channelMean[1]/255.0, channelMean[2]/255.0))


def saveMean(fname, data):
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = '3'
    et.SubElement(root, 'Row').text = str(imgSize)
    et.SubElement(root, 'Col').text = str(imgSize)
    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(meanImg, 'rows').text = '1'
    et.SubElement(meanImg, 'cols').text = str(imgSize * imgSize * 3)
    et.SubElement(meanImg, 'dt').text = 'f'
    et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(data, (imgSize * imgSize * 3))])

    tree = et.ElementTree(root)
    tree.write(fname)
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = '  '))


def saveTrainImages(filename, foldername):
    train_foldername = os.path.join(os.path.abspath(foldername), 'Train')
    if not os.path.exists(train_foldername):
        os.makedirs(train_foldername)
    data = {}
    dataMean = np.zeros((3, imgSize, imgSize))  # mean is in CHW format.
    with open(os.path.join(foldername, 'train_map.txt'), 'w') as mapFile:
        with open(os.path.join(foldername, 'train_regrLabels.txt'), 'w') as regrFile:
            for ifile in range(1, 6):
                with open(os.path.join(foldername, 'cifar-10-batches-py', 'data_batch_' + str(ifile)), 'rb') as f:
                    data = cp.load(f, encoding='latin1')
                    for i in range(10000):
                        fname = os.path.join(
                            train_foldername, ('%05d.png' % (i + (ifile - 1) * 10000)))
                        saveImage(
                            fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 4, mean=dataMean)
    dataMean = dataMean / (50 * 1000)
    saveMean(os.path.join(foldername, 'CIFAR-10_mean.xml'), dataMean)


def saveTestImages(filename, foldername):
    test_foldername = os.path.join(os.path.abspath(foldername), 'Test')
    if not os.path.exists(test_foldername):
        os.makedirs(test_foldername)

    with open(os.path.join(foldername, 'test_map.txt'), 'w') as mapFile:
        with open(os.path.join(foldername, 'test_regrLabels.txt'), 'w') as regrFile:
            with open(os.path.join(foldername, 'cifar-10-batches-py', 'test_batch'), 'rb') as f:
                data = cp.load(f, encoding='latin1')
                for i in range(10000):
                    fname = os.path.join(test_foldername, ('%05d.png' % i))
                    saveImage(fname, data['data'][i, :],
                              data['labels'][i], mapFile, regrFile, 0)


url_cifar_data = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
data_dir = './data/CIFAR-10'
data_filename = data_dir + '/cifar-10-python.tar.gz'
train_filename = data_dir + '/Train_cntk_text.txt'
test_filename = data_dir + '/Test_cntk_text.txt'


root_dir = os.getcwd()

os.makedirs(data_dir, exist_ok=True)


trn, tst = loadData(url_cifar_data, data_filename)
print("Writing train text file...")
saveTxt(train_filename, trn)
print("Done.")
print("Writing test text file...")
saveTxt(test_filename, tst)
print("Done.")
print('Converting train data to png images...')
saveTrainImages(train_filename, data_dir)
print("Done.")
print('Converting test data to png images...')
saveTestImages(test_filename, data_dir)
print("Done.")
