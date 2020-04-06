import numpy as np
from gensim.models import Word2Vec
import pickle
import pandas as pd
import collections
import os
from tqdm import tqdm
import logging
from sortedcontainers import SortedSet
import itertools
from pathlib import Path

def setup_logging():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


setup_logging()

Rating = collections.namedtuple("Rating", "movie_id, rating, date")
path = "/home/p.korobov/data/netflix/processed"


df = None
for i, file in enumerate(os.listdir(path)):
    
    logging.info('Reading file #{}'.format(i))
    if df is None:
        df = pd.read_pickle(path + "/" + file)
        if type(df) is not pd.core.frame.DataFrame:
            df = None
    else:
        try:
            df = pd.concat([df, pd.read_pickle(Path(path) / file)])
        except:
            logging.info("Error!")


movies_sequences = df['ratings'].apply(lambda x: [*map(lambda y: str(y[0]), x)])

s = set()
for l in movies_sequences:
    s |= set(l)
s = sorted(np.array(list(s)).astype(np.int32))

to_new_index = dict(zip(s, range(len(s))))
to_old_index = dict(zip(range(len(s)), s))

path = "/home/p.korobov/data/netflix/matrix_env"
with open(Path(path) / "to_old_index.pkl", "wb") as file:
    pickle.dump(to_old_index, file)

with open(Path(path) / "to_new_index.pkl", "wb") as file:
    pickle.dump(to_new_index, file)

# iter = 20
model = Word2Vec(sentences=movies_sequences, size=20, window=20, min_count=1, workers=30, sg=1, iter=20, compute_loss=True)

embeddings = dict()
g = lambda x: model.wv[str(x)]
for elem in s:
    embeddings[to_new_index[elem]] = np.array(g(elem))
max_norm = np.linalg.norm([*embeddings.values()], ord=np.inf, axis=1).max()

for elem in s:
    embeddings[to_new_index[elem]] /= max_norm
with open(Path(path) / "embeddings_dict.pkl", "wb") as file:
    pickle.dump(embeddings, file)
logging.info("Embeddings have been learned")

N = len(s)
W = np.zeros((N, N))
counts = np.zeros((N, N))


def binarize_rating(x):
    if x > 3:
        return 1.0
    elif x == 3:
        return 0.5
    return 0.0


next_ratings_window = 25
good_ratings_cnt = 0
all_ratings_cnt = 0

for _, user_ratings in tqdm(df['ratings'].iteritems()):

    good_ratings_cnt += np.sum([*map(lambda x: binarize_rating(x.rating), user_ratings)])
    all_ratings_cnt += len(user_ratings)

    dates_set = SortedSet([elem.date for elem in user_ratings])
    grouped_ratings = [[rating for rating in user_ratings if rating.date == val] for val in dates_set]

    for i, elem in enumerate(grouped_ratings[:-1]):
        current_ratings = grouped_ratings[i].copy()
        next_ratings = []
        for k in range(i + 1, min(len(grouped_ratings), i + 1 + next_ratings_window)):
            next_ratings += grouped_ratings[k]
        for cur_rating, next_rating in itertools.product(current_ratings, next_ratings):
            assert cur_rating.date.date() < next_rating.date.date()
            W[to_new_index[cur_rating.movie_id], to_new_index[next_rating.movie_id]] += binarize_rating(next_rating.rating)
            counts[to_new_index[cur_rating.movie_id], to_new_index[next_rating.movie_id]] += 1

W_1 = (W + good_ratings_cnt) / (counts + all_ratings_cnt)
with open("/home/p.korobov/data/netflix/matrix_env/W_matrix.pkl", "wb") as file:
    pickle.dump(W_1, file)

W_2 = (W + 1) / (counts + 2)
with open("/home/p.korobov/data/netflix/matrix_env/W_matrix_2.pkl", "wb") as file:
    pickle.dump(W_2, file)
    
logging.info("W has been constructed")
