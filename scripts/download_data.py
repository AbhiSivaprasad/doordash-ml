from src.data.download import download, DownloadArgs


if __name__ == "__main__":
    download(args=DownloadArgs().parse_args())
