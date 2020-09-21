import zipfile
import gdown
import shutil, os, json
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def list_files(folder, pattern):
    return list(Path(folder).glob(pattern))

def unzip(file, output_dir):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        
def gdrive_down(file_id, output_file):
    url = 'https://drive.google.com/uc?id=' + file_id
    gdown.download(url, output_file, quiet=False)
    
def move_files(source, destination, pattern='*'): 
    dest = Path(destination)
    
    if not dest.is_dir:
        print('Create folder: '+ str(dest))
        os.mkdir(dest)
        
    print('Destination path: ' + str(dest))
    list_of_files = list(Path(source).glob(pattern))
    print('Copy {} files'.format(len(list_of_files)))
    for path in list_of_files:
        print('Moving ' + str(path))
        shutil.copy(path, dest)
        
    print('Done copying')
    
def read_json(fp):
    with open(fp) as f:
        data = json.load(f)

    return data

def write_json(fp, data):
    with open(fp, 'w') as f:
        json.dump(data, f)
        
def imshow(img, size=15):
    plt.figure(figsize = (size,size))
    plt.axis('off')
    plt.imshow(img)
    
def imread_show(file_path, size=15):
    im = cv2.imread(str(file_path))
    imshow(im)