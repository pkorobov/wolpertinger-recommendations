import argparse
import json
import re
import collections
import tqdm
import multiprocessing
import hashlib
import os
import logging
import pickle

import dateutil as du
import numpy as np
import scipy.sparse as ss
import pandas as pd


N_MOVIES = 17770
N_USERS_WITH_MISSING = 2649429
N_USERS = 480189
N_RATINGS = 100480507

MOVIE_INDEX_REGEX = re.compile("(\\d+):")


Rating = collections.namedtuple("Rating", "movie_id, rating, date")


def setup_logging():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


def write_data_to_partitioned_dataframes(data_file_paths, output_dir, n_partitions, n_cores=2):
    ratings_coo, dates_coo, movie_frequency = _load_coo_with_movie_frequency(data_file_paths)
    assert len(movie_frequency) == N_MOVIES
    pickle.dump(movie_frequency, open(os.path.join(output_dir, "movie_frequency.pkl"), "wb"))
    
    ratings_csr, rating_user_ids = _coo_to_compact_csr(ratings_coo)
    dates_csr, date_user_ids = _coo_to_compact_csr(dates_coo)
    assert np.alltrue(dates_csr.indptr == ratings_csr.indptr)
    assert np.alltrue(dates_csr.indices == ratings_csr.indices)
    
    assert np.alltrue(rating_user_ids == date_user_ids)
    user_ids = rating_user_ids
    assert len(user_ids) == N_USERS
    
    _matrices_to_dataframe_partitions(user_ids, ratings_csr, dates_csr, output_dir, n_cores=n_cores, n_partitions=n_partitions)
    
    
def _load_coo_with_movie_frequency(data_file_paths):
    logging.info("Loading raw data from: %s", data_file_paths)
    
    movie_frequency = collections.Counter()
    
    ratings = np.zeros(N_RATINGS, dtype=np.float)
    dates = np.zeros(N_RATINGS, dtype=np.int64)
    i = np.zeros(N_RATINGS, dtype=np.int64)
    j = np.zeros(N_RATINGS, dtype=np.int64)

    movie_index = None
    rating_index = 0
    for data_file_path in data_file_paths:
        with open(data_file_path) as data_file:
            for line in tqdm.tqdm(data_file):
                movie_index_if_any = _parse_movie_index_line(line)

                if movie_index_if_any is not None:
                    movie_index = movie_index_if_any
                    continue

                user_index, rating, date_int = _parse_rating_line(line)
                ratings[rating_index] = rating
                dates[rating_index] = date_int
                i[rating_index] = user_index
                j[rating_index] = movie_index
                rating_index += 1
                
                movie_frequency[movie_index] += 1
                
    ratings_coo = ss.coo_matrix((ratings, (i, j)), shape=(N_USERS_WITH_MISSING + 1, N_MOVIES + 1))
    dates_coo = ss.coo_matrix((dates, (i, j)), shape=(N_USERS_WITH_MISSING + 1, N_MOVIES + 1))
    
    return ratings_coo, dates_coo, movie_frequency


def _parse_movie_index_line(line):
    match = re.match(MOVIE_INDEX_REGEX, line)
    if match is None:
        return
    movie_index = int(match.group(1))
    return movie_index


def _parse_rating_line(line):
    rating_line_parts = line.split(",")
    rating = float(rating_line_parts[1])
    date_int = _date_str_to_int(rating_line_parts[2])
    user_index = int(rating_line_parts[0])
    return user_index, rating, date_int


def _date_str_to_int(date_str):
    return int(date_str.replace("-", ""))


def _coo_to_compact_csr(coo):
    csr = coo.tocsr()
    
    assert np.sum(csr.getnnz(axis=1) > 0).sum() == N_USERS, np.sum(csr.getnnz(axis=1) > 0).sum()
    assert np.sum(csr.getnnz(axis=0) > 0).sum() == N_MOVIES, np.sum(csr.getnnz(axis=1) > 0).sum()
    
    nonzero_user_ind = np.arange(csr.shape[0], dtype=int)[csr.getnnz(axis=1) > 0]
    
    return csr[nonzero_user_ind], nonzero_user_ind


def _matrices_to_dataframe_partitions(user_ids, ratings_csr, dates_csr, save_dir, n_partitions=40, n_cores=1):
    logging.info("Started converting csr to data frames")
    
    indices = np.arange(len(user_ids))
    data_chunks = [(user_ids[chunk_indices], ratings_csr[chunk_indices], dates_csr[chunk_indices], save_dir) for chunk_indices in np.array_split(indices, n_partitions)]
    
    if n_cores > 1:
        pool = multiprocessing.Pool(n_cores)
        partition_file_paths = pool.map(_matrices_tuple_to_dataframe, data_chunks)
        pool.close()
        pool.join()
    else:
        partition_file_paths = list(map(_matrices_tuple_to_dataframe, data_chunks))
        
    return partition_file_paths


def _matrices_tuple_to_dataframe(arguments):
    user_ids, ratings_csr, dates_csr, save_dir = arguments
    return _matrices_to_dataframe(user_ids, ratings_csr, dates_csr, save_dir)


def _matrices_to_dataframe(user_ids, ratings_csr, dates_csr, save_dir):
    assert np.alltrue(ratings_csr.indptr == dates_csr.indptr)
    assert np.alltrue(ratings_csr.indices == dates_csr.indices)
    
    m = hashlib.md5()
    m.update(str(sorted(user_ids.tolist())).encode("UTF-8"))
    partition_file_path = os.path.join(save_dir, "partition-" + m.hexdigest() + ".pkl")
    
    if os.path.exists(partition_file_path) and os.path.getsize(partition_file_path) > 0:
        logging.info("Skipping %s because it already exists", partition_file_path)
        return partition_file_path
    
    user_ratings = []
    for i in range(len(user_ids)):
        start_ind, end_ind = ratings_csr.indptr[i], ratings_csr.indptr[i+1]

        movie_ids = ratings_csr.indices[start_ind: end_ind]
        ratings = ratings_csr.data[start_ind: end_ind]
        dates = [du.parser.parse(str(d)) for d in dates_csr.data[start_ind: end_ind]]

        rating_triplets = sorted([Rating(*triplet) for triplet in zip(movie_ids, ratings, dates)], key=lambda r: (r.date, r.movie_id))

        user_ratings.append({
            "user_id": user_ids[i],
            "ratings": rating_triplets
        })
    
    pd.DataFrame(user_ratings).set_index("user_id").to_pickle(partition_file_path)
    logging.info("Written %s", partition_file_path)
    return partition_file_path


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", help="Directory where experiment files are located", type=str, required=True)
    parser.add_argument("--cores", help="Number of cores for parallel data writing", type=int, required=False, default=2)
    parser.add_argument("experiment", help="Experiment name", type=str)
    args = parser.parse_args()

    logging.info("Preprocessing data for experiment " + args.experiment)

    config = json.load(open(os.path.join(args.experiment_dir, args.experiment, "config.json")))

    raw_data_dir = config["input"]["raw"]["dir"]
    data_file_paths = [os.path.join(raw_data_dir, raw_data_file) for raw_data_file in os.listdir(raw_data_dir) if raw_data_file.startswith(config["input"]["raw"]["prefix"])]
    logging.info("Found raw data files: " + ",".join(data_file_paths))

    processed_files_config = config["input"]["processed"]
    processed_files_dir = processed_files_config["dir"]
    movie_frequency_file_path = os.path.join(processed_files_dir, "movie_frequency.pkl")
    processed_files_partitions = processed_files_config["partitions"]

    processed_files_paths = [os.path.join(processed_files_dir, file_name) for file_name in os.listdir(processed_files_dir) if file_name.startswith(processed_files_config["prefix"])]
    if len(processed_files_paths) != processed_files_partitions or not os.path.exists(movie_frequency_file_path):
        write_data_to_partitioned_dataframes(data_file_paths, processed_files_dir, n_partitions=processed_files_partitions, n_cores=args.cores)
    else:
        logging.info("It seems that all the preprocessed files are already in " + processed_files_dir)


if __name__ == "__main__":
    main()
