import hashlib


def hash_string(input_string: str, digits: int = 8):
        return int(hashlib.sha1(input_string.encode('utf-8')).hexdigest(), 16) % (10 ** digits)
