import cv2
import wandb
import boto3
import requests 
import urllib

from os import listdir
from os.path import join
from functools import partial
from typing import List
from tempfile import TemporaryDirectory
import multiprocessing.pool as mpp
from tqdm import tqdm

from .utils import hash_string


def upload_directory_to_s3(s3_client, 
                           bucket_name, 
                           object_prefix: str, 
                           dirpath: str, 
                           filenames: List[str], 
                           num_workers: int = 16):
    """Upload a directory w/o subdirectories to s3"""
    filepaths = [join(dirpath, filename) for filename in filenames]
    object_names = [join(object_prefix, filename) for filename in filenames]

    # define function instead of partial since positional args needed for map
    def thread_f(filepath, key):
        return s3_client.upload_file(filepath, bucket_name, key)

    # upload in parallel with a thread pool
    mpp.Pool.istarmap = istarmap
    for _ in tqdm(
        mpp.ThreadPool(num_workers).istarmap(thread_f, zip(filepaths, object_names)), 
        total=len(filenames)
    ):
        pass


def download_images_from_urls(urls: List[str], dirpath: str, filenames: List[str], num_workers: int = 16):
    """Download images to a local dir""" 
    # file name is a hash of the url
    filepaths = [join(dirpath, filename) for url, filename in zip(urls, filenames)]

    # download in parallel with a thread pool
    mpp.Pool.istarmap = istarmap
    for _ in tqdm(mpp.ThreadPool(num_workers).istarmap(download_image, zip(urls, filepaths)), total=len(urls)):
        pass


def download_image(url: str, filepath: str):
    """Download image from url to filepath"""
    # already doing python multiprocessing, so prob better to not double up
    cv2.setNumThreads(0)

    # fetch image
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=5)

    # process image
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    scaling_factor = 1024.0 / max(height, width)
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
    # write image
    cv2.imwrite(filepath, img)

    # move image (make sure completely downloaded)
    if os.stat(filepath).st_size > 0:
        raise ValueError('File is empty', filepath)


def download_image_old(url: str, filepath: str):
    """Download image from url to filepath"""
    response = requests.get(url)

    if not response.ok:
        print(response)
        raise ValueError("Failed Request:", url)

    img_data = response.content 
    with open(filepath, 'wb') as f:
        f.write(img_data)


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
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


if __name__ == '__main__':
    s3_client = boto3.client('s3')

    urls = [
        "http://cdn.doordash.com/media/photos/eef3de6b-c3a0-42c1-9903-9673c3fbe0e6-retina-large-jpeg",
        "http://cdn.doordash.com/media/photos/1737dee7-af1f-4530-89ea-f978af01530e-retina-large-jpeg",
        "http://cdn.doordash.com/media/photos/596e6370-be62-43f2-be7a-cdffa7ff9d70-retina-large-jpeg"
    ]

    download_images_to_bucket(s3_client, "glisten-images", "train/alcohol", urls, [str(hash(url)) for url in urls])
    # upload_directory_to_s3(s3_client, "glisten-images", "train/baby", "tmp")
