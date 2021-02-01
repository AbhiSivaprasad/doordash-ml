import cv2
import wandb
import boto3
import requests 
import urllib
import hashlib

import numpy as np
import multiprocessing.pool as mpp

from os import listdir, stat, walk
from os.path import join, relpath
from functools import partial
from typing import List
from tempfile import TemporaryDirectory
from tqdm import tqdm
from pathlib import Path


def upload_directory_to_s3(s3_client, 
                           bucket_name, 
                           dirpath: str, 
                           object_prefix: str,
                           filenames: List[str], 
                           num_workers: int = 16):
    """Upload a directory recursively to s3"""
    # recursively collect all filepaths
    filepaths = [join(root, f) for root, dirs, files in walk(dirpath) for f in files]
    object_names = [relpath(filepath, dirpath) for filepath in filepaths]
    object_keys = [join(object_prefix, object_name) for object_name in object_names]

    # define function instead of partial since positional args needed for map
    def thread_f(filepath, key):
        return s3_client.upload_file(filepath, bucket_name, key)

    # upload in parallel with a thread pool
    mpp.Pool.istarmap = istarmap
    for _ in tqdm(
        mpp.ThreadPool(num_workers).istarmap(thread_f, zip(filepaths, object_keys)), 
        total=len(filenames)
    ):
        pass


def download_images_from_urls(urls: List[str], dirpath: str, filenames: List[str], num_workers: int = 64):
    """Download images to a local dir""" 
    filepaths = []
    hash_dirs = set()

    # file name is a hash of the url
    for url, filename in zip(urls, filenames):
        # strip ".jpeg"
        stripped_filename = remove_extension(filename) 
        assert(stripped_filename is not None)

        # first 3 characters of hash
        hash_dir = hashlib.sha1(stripped_filename.encode('utf-8')).hexdigest()[:2]
        hash_dirs.add(hash_dir)

        # file path
        filepaths.append(join(dirpath, hash_dir, filename))

    # create all the hash dirs
    for hash_dir in hash_dirs:
        Path(join(dirpath, hash_dir)).mkdir(exist_ok=True)

    # download in parallel with a thread pool
    mpp.Pool.istarmap = istarmap
    bad_urls = [
        url for url in tqdm(
            mpp.ThreadPool(num_workers).istarmap(download_image, zip(urls, filepaths)), 
            total=len(urls)
        ) if url is not None
    ]        
    
    return bad_urls


def download_image(url: str, filepath: str, tries: int = 3):
    """Download image from url to filepath"""
    print(f"Downloading from {url}")

    # already doing python multiprocessing, so prob better to not double up
    cv2.setNumThreads(0)

    # fetch image
    try: 
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        # try again
        if tries > 0:
            print(f"Retrying: {url}")
            return download_image(url, filepath, tries - 1)
        else:
            print(f"Failed Request: {url}")
            print(e)
            return url

    # process image
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if img is None:
        # try again
        if tries > 0:
            print(f"Retrying: {url}")
            return download_image(url, filepath, tries - 1)
        else:
            print(f"Failed Request: {url}")
            return url

    height, width = img.shape[:2]
    scaling_factor = 1024.0 / max(height, width)
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
    # write image
    print(f"Writing image to {filepath}")
    if not cv2.imwrite(filepath, img):
        return Exception("could not write image")

    # move image (make sure completely downloaded)
    if stat(filepath).st_size == 0:
        # try again
        if tries > 0:
            print(f"Retrying: {url}")
            return download_image(url, filepath, tries - 1)
        else:
            raise ValueError('File is empty', filepath)

    print(f"{filepath} exists")
    return None


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap"""
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


def remove_extension(filename: str):
    stripped_filename = ".".join(filename.split(".")[:-1])

    # if the stripped name is "" then extension not found
    return stripped_filename if stripped_filename != "" else None


if __name__ == '__main__':
    s3_client = boto3.client('s3')

    urls = [
        "http://cdn.doordash.com/media/photos/eef3de6b-c3a0-42c1-9903-9673c3fbe0e6-retina-large-jpeg",
        "http://cdn.doordash.com/media/photos/1737dee7-af1f-4530-89ea-f978af01530e-retina-large-jpeg",
        "http://cdn.doordash.com/media/photos/596e6370-be62-43f2-be7a-cdffa7ff9d70-retina-large-jpeg"
    ]

    download_images_to_bucket(s3_client, "glisten-images", "train/alcohol", urls, [str(hash(url)) for url in urls])
    # upload_directory_to_s3(s3_client, "glisten-images", "train/baby", "tmp")
