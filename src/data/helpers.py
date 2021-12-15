import pathlib


def create_dir(path: pathlib.PosixPath) -> pathlib.PosixPath:
    path.mkdir(exist_ok=True, parents=True)
    return path

