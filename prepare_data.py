import numpy as np
from pyarrow import parquet as pq
from gensim.models import Word2Vec
import pickle
import pandas as pd
import collections
import os
from tqdm import tqdm
import logging
from sortedcontainers import SortedSet
import itertools


def setup_logging():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


setup_logging()

Rating = collections.namedtuple("Rating", "movie_id, rating, date")
path = "/Users/p.korobov/data/netflix/processed"

MAX_W_SIZE = 10000

for i, file in enumerate(os.listdir(path)):
    logging.info('Reading file #{}'.format(i))
    if i == 0:
        df = pd.read_pickle(path + "/" + file)
    else:
        try:
            df = pd.concat([df, pd.read_pickle(path + "/" + file)])
        except:
            logging.info("Error!")
movies_sequences = df['ratings'].apply(lambda x: [*map(lambda y: str(y[0]), x)])

s = set()
for l in movies_sequences:
    s |= set(l)
s = sorted(np.array(list(s)).astype(np.int32))

to_new_index = dict(zip(s, range(len(s))))
to_old_index = dict(zip(range(len(s)), s))

with open("to_old_index.pkl", "wb") as file:
    pickle.dump(to_old_index, file)

with open("to_new_index.pkl", "wb") as file:
    pickle.dump(to_new_index, file)

model = Word2Vec(sentences=movies_sequences, size=20, window=20, min_count=1, workers=4, sg=1, iter=20)

embeddings = dict()
g = lambda x: model.wv[str(x)]
for elem in s:
    embeddings[to_new_index[elem]] = np.array(g(elem))
max_norm = np.linalg.norm([*embeddings.values()], ord=np.inf, axis=1).max()

for elem in s:
    embeddings[to_new_index[elem]] /= max_norm

N = min(len(s), MAX_W_SIZE)
W = np.zeros((N, N))
counts = np.zeros((N, N))


def binarize_rating(x):
    if x.rating > 3:
        return 1.0
    elif x.rating == 3:
        return 0.5
    return 0.0


next_ratings_window = 10
good_ratings_cnt = 0
all_ratings_cnt = 0

for _, user_ratings in tqdm(df['ratings'].iteritems()):

    good_ratings_cnt += map(lambda x: binarize_rating(x.rating), user_ratings)
    all_ratings_cnt += len(user_ratings)

    dates_set = SortedSet([elem.date for elem in user_ratings])
    grouped_ratings = [[rating for rating in user_ratings if rating.date == val] for val in dates_set]

    for i, elem in enumerate(grouped_ratings[:-1]):
        current_ratings = grouped_ratings[i].copy()
        next_ratings = []
        for k in range(i + 1, min(len(grouped_ratings), i + 1 + 10)):
            next_ratings += grouped_ratings[k]
        for cur_rating, next_rating in itertools.product(current_ratings, next_ratings):
            assert cur_rating.date.date() < next_rating.date.date()
            W[cur_rating.movie_id, next_rating.movie_id] += binarize_rating(next_rating.rating)
            counts[cur_rating.movie_id, next_rating.movie_id] += 1

# W = (W + 1) / (counts + 2)
W = (W + 1) / (counts + all_ratings_cnt / good_ratings_cnt)

with open("W_matrix.pkl", "wb") as file:
    pickle.dump(W, file)

with open("embeddings_dict.pkl", "wb") as file:
    pickle.dump(embeddings, file)
