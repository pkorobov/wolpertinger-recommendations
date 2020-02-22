import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def months_between(date1, date2):
    return int((date2 - date1).days / 31)
