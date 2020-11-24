import boto3
import requests 

from os import listdir
from os.path import join
from functools import partial
from typing import List
from tempfile import TemporaryDirectory
from multiprocessing.pool import ThreadPool

from .utils import hash_string


def download_images_to_bucket(s3_client, bucket_name: str, object_prefix: str, urls: List[str]):
    """Download images from urls to s3 bucket"""
    # create temp directory for download
    temp_dir = TemporaryDirectory()
    temp_dirpath = temp_dir.name

    # download images from urls into dir
    download_images(urls, temp_dirpath)

    # upload directory of images
    upload_directory_to_s3(s3_client, bucket_name, object_prefix, temp_dirpath)


def upload_directory_to_s3(s3_client, bucket_name, object_prefix: str, dirpath: str, num_workers: int = 32):
    """Upload a directory w/o subdirectories to s3"""
    filenames = [filename for filename in listdir(dirpath)]
    filepaths = [join(dirpath, filename) for filename in filenames]
    object_names = [join(object_prefix, filename) for filename in filenames]

    # define function instead of partial since positional args needed for map
    def thread_f(filepath, key):
        return s3_client.upload_file(filepath, bucket_name, key)

    # upload in parallel with a thread pool
    ThreadPool(num_workers).starmap(thread_f, zip(filepaths, object_names))


def download_images(urls: List[str], dirpath: str, num_workers: int = 32):
    """Download images to a local dir""" 
    # file name is a hash of the url
    filepaths = [join(dirpath, f"{hash_string(url)}.jpeg") for url in urls]

    # download in parallel with a thread pool
    ThreadPool(num_workers).starmap(download_image, zip(urls, filepaths))
     

def download_image(url: str, filepath: str):
    """Download image from url to filepath"""
    response = requests.get(url)

    if not response.ok:
        raise ValueError("Failed Request:", url)

    img_data = response.content 
    with open(filepath, 'wb') as f:
        f.write(img_data)


if __name__ == '__main__':
    s3_client = boto3.client('s3')

    urls = [
        "http://cdn.doordash.com/media/photos/1737dee7-af1f-4530-89ea-f978af01530e-retina-large-jpeg",
        "http://cdn.doordash.com/media/photos/596e6370-be62-43f2-be7a-cdffa7ff9d70-retina-large-jpeg"
    ]

    # download_images_to_bucket(s3_client, "glisten-images", "train/alcohol", urls)
    upload_directory_to_s3(s3_client, "glisten-images", "train/baby", "tmp")
