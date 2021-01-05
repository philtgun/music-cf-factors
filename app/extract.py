from __future__ import annotations

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from implicit.approximate_als import AlternatingLeastSquares
from implicit.datasets.million_song_dataset import get_msd_taste_profile
from implicit.nearest_neighbours import bm25_weight


def extract(factors_file: str, songs_file: str):
    model = AlternatingLeastSquares(factors=64, dtype=np.float32)
    songs, users, plays = get_msd_taste_profile()

    logging.info("Saving songs")
    np.save(songs_file, songs[:, 0].astype("U"))

    logging.info("Weighting matrix by bm25_weight")
    # values taken from implicit.examples
    plays = bm25_weight(plays, K1=100, B=0.8)

    logging.info("Compressing matrix")
    plays = plays.tocsr()

    logging.info("Training model")
    model.fit(plays)

    logging.info("Saving item factors")
    factors = model.item_factors
    np.save(factors_file, factors)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--output-factors",
        default="data/factors.npy",
        help="NPY file with song factors as a matrix",
    )
    parser.add_argument(
        "--output-songs",
        default="data/songs.npy",
        help="NPY file with the list of MSD song ids",
    )
    args = parser.parse_args()
    extract(args.output_factors, args.output_songs)
