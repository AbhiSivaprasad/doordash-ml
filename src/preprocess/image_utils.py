import boto3
import requests 

from os import listdir
from os.path import join
from typing import List
from tempfile import TemporaryDirectory
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


def upload_directory_to_s3(s3_client, bucket_name, object_prefix: str, dirpath: str):
    """Upload a directory w/o subdirectories to s3"""
    for filename in listdir(dirpath):
        filepath = join(dirpath, filename)
        object_name = join(object_prefix, filename)
        s3_client.upload_file(filepath, bucket_name, object_name)


def download_images(urls: List[str], dirpath: str):
    """Download images to a local dir""" 
    for url in urls:
        filename = f"{hash_string(url)}.jpeg"
        filepath = join(dirpath, filename)
        download_image(url, filepath)
     

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

    download_images_to_bucket(s3_client, "glisten-images", "train/alcohol", urls)
