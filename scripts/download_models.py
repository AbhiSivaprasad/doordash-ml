from src.models.download import download, DownloadArgs


if __name__ == "__main__":
    download(args=DownloadArgs().parse_args(known_only=True))
