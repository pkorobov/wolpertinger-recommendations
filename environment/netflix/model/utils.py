import os

import dateutil as du


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def months_between(date1, date2):
    r = du.relativedelta.relativedelta(date2, date1)
    return r.months + (12 * r.years)
