from os.path import exists, dirname, relpath, join
from os import makedirs


def download_directory_from_s3(bucket_resource, s3_folder: str, write_dir: str):
    Path(write_dir).mkdir(parent=True, exist_ok=True)
    for obj in bucket_resource.objects.filter(Prefix=s3_folder):
        # relative s3 path with respect to s3_folder added to write_dir
        local_path = join(write_dir, relpath(obj.key, s3_folder))

        # make parent directory if it doesn't exist
        local_dir = dirname(local_path)
        Path(local_dir).mkdir(parent=True, exist_ok=True)

        # download to local dir
        bucket_resource.download_file(obj.key, local_path)
