__author__ = 'skin.ly'
import pandas as pd
from isic_api import ISICApi
from pandas.io.json import json_normalize
from os.path import join
import multiprocessing as mp
from tqdm import tqdm
from os.path import exists
import argparse
import numpy as np
from shutil import copyfile
from os import mkdir


def get_image_details(image, api):
    image_detail = api.getJson('image/{}'.format(image['_id']))
    return pd.Series(image_detail)

def load_meta_data(image_list, api):
    meta = pd.DataFrame()
    for image in tqdm(image_list, desc="getting meta-data"):
        image_details = api.getJson('image/{}'.format(image['_id']))
        meta = meta.append(json_normalize(image_details, sep='_'), ignore_index=True, sort=True)
    return meta

def download_image(image, save_dir, api):
    imageFileOutputPath = join(save_dir, '%s.jpg' % image['name'])
    if exists(imageFileOutputPath):
        return
    imageFileResp = api.get('image/%s/download' % image['_id'])
    imageFileResp.raise_for_status()
    with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
        for chunk in imageFileResp:
            imageFileOutputStream.write(chunk)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--account_user', required=True,
                        help="Create account at https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main")
    parser.add_argument('--account_password', required=True)
    parser.add_argument('--out_dir', required=True, help="directory where dataset will be stored")
    options = parser.parse_args()


    ### API authentication
    api = ISICApi(username=options.account_user, password=options.account_password)

    ### get list of images
    imageList = api.getJson('image?limit=100000&offset=0&sort=name')
    print('Got list of {} images'.format(len(imageList)))

    ### download images and save metadata
    out_dir = options.out_dir
    for subfolder in ['images', 'dataset', 'meta']:
        try:
            mkdir(join(out_dir, subfolder))
        except FileExistsError:
            print('{} already exists'.format(subfolder))
    for subfolder in ['case', 'control']:
        try:
            mkdir(join(out_dir, 'dataset', subfolder))
        except FileExistsError:
            print('{} already exists'.format(subfolder))

    ### download images
    pool = mp.Pool(processes=mp.cpu_count())
    f_list = []
    for image in tqdm(imageList, desc="downloading dataset"):
        f_list.append(pool.apply_async(download_image(image, join(out_dir, 'images'), api)))


    ### get metadata of images
    # meta_data = load_meta_data(imageList, api)
    # meta_data.to_pickle(join(out_dir, 'meta', 'metadata.pkl'))
    # meta_data.to_csv(join(out_dir, 'meta', 'metadata.csv'), index=False)
    meta_data = pd.read_pickle(join(out_dir, 'meta', 'metadata.pkl'))

    ### Identify malignant images and download 1x malignant and 2x benign images
    malignant = meta_data[meta_data.meta_clinical_benign_malignant == 'malignant']._id.values
    benign = meta_data[meta_data.meta_clinical_benign_malignant == 'benign']._id.values
    np.random.seed(0)
    benign_sample = np.random.choice(benign, 2 * malignant.shape[0], replace=False)
    accept_ids = np.concatenate([malignant, benign_sample])

    ### get list of images to be used as case/controls
    imageList = [var for var in imageList if var['_id'] in accept_ids]
    for image in tqdm(imageList, desc="splitting images in case/control groups"):
        f_name = "%s.jpg" % image['name']
        f_path = join(out_dir, 'images', f_name)
        is_malignant = True if image['_id'] in malignant else False
        save_to = join(out_dir, 'dataset', 'case' if is_malignant else 'control')
        copyfile(f_path, join(save_to, f_name))

if __name__ == '__main__':
    main() 