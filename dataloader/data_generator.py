import sys
import os
#import urllib.request   ## python 3
from urllib import urlretrieve ## python 2
import tarfile
import zipfile
from preprocess import cifar_preprocess, svhn_preprocess
import numpy as np

def download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r {0:.1%} already downloaded".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

            
def download(data_type):
    url_dict={'cifar-10':'http://www.cs.toronto.edu/'
                    '~kriz/cifar-10-python.tar.gz',
                    'cifar-100': 'http://www.cs.toronto.edu/'
                    '~kriz/cifar-100-python.tar.gz',
                    'svhn_train': "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                    'svhn_test': "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                    'svhn_extra': "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"}
    url=url_dict[data_type]
    file_name=url.split('/')[-1]
    data_folder='./'+data_type
    if data_type=='cifar-10':
        data_path=os.path.join(data_folder, data_type+'-batches-py')
    elif data_type=='cifar-100':
        data_path=os.path.join(data_folder, data_type+'-python')
    elif data_type=='svhn_train' or data_type=='svhn_test' or data_type=='svhn_extra':
        data_path=data_folder
        
    if os.path.exists(data_path)==False:
        
        os.mkdir(data_folder)
        file_path=os.path.join(data_folder, file_name)
        
        print "Downloading data file from %s to %s" % (url, file_path)
        urlretrieve(url=url, 
                                 filename=file_path,
                                 reporthook=download_progress)
        
        print "\nExtracting file..."
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode='r').extractall(data_folder)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(data_folder)
        
        print "Successfully downloaded and extracted"

    else:
        print "Data file already exists"
    
    return data_path

    
def data_normalization(train_data_raw, test_data_raw, normalize_type):
    if normalize_type=='divide-255':
        train_data=train_data_raw/255.0
        test_data=test_data_raw/255.0
        
        return train_data, test_data
    elif normalize_type=='divide-256':
        train_data=train_data_raw/256.0
        test_data=test_data_raw/256.0
        
        return train_data, test_data
    elif normalize_type=='by-channels':
        train_data=np.zeros(train_data_raw.shape)
        test_data=np.zeros(test_data_raw.shape)
        for channel in range(train_data_raw.shape[-1]):
            images=np.concatenate((train_data_raw, test_data_raw), axis=0)
            channel_mean=np.mean(images[:,:,:,channel])
            channel_std=np.std(images[:,:,:,channel])
            train_data[:,:,:,channel]=(train_data_raw[:,:,:,channel]-channel_mean)/channel_std
            test_data[:,:,:,channel]=(test_data_raw[:,:,:,channel]-channel_mean)/channel_std
        
        return train_data, test_data
    
    elif normalize_type=='None':
        
        return train_data_raw, test_data_raw
   
def load_data(data_type, normalize_type):  ##  cifar-10  or cifar-100
    if data_type=='svhn':
        data_path_train=download(data_type+'_train')
        data_path_extra=download(data_type+'_extra')
        data_path_test=download(data_type+'_test')
        train_data_raw, train_labels, test_data_raw, test_labels=svhn_preprocess(data_path_train, data_path_extra, data_path_test)
    else:
        data_path=download(data_type)
        train_data_raw, train_labels, test_data_raw, test_labels=cifar_preprocess(data_type, data_path)
    
    train_data, test_data=data_normalization(train_data_raw, test_data_raw, normalize_type)
    
    return train_data, train_labels, test_data, test_labels
    


