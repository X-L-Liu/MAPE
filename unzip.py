import zipfile

if __name__ == '__main__':  
    with zipfile.ZipFile('SAU_Net/cifar10/SAU_Net_MAPE_0.9594.zip') as zip_file:
        zip_file.extractall('SAU_Net/cifar10')
