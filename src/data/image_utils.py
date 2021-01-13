import hashlib


def get_image_hashdir(image_name: str):
    stripped_image_name = ".".join(image_name.split(".")[:-1])

    # if the stripped name is "" then extension not found
    assert stripped_image_name != ""

    # first two chars of hash
    return hashlib.sha1(stripped_image_name.encode('utf-8')).hexdigest()[:2]
