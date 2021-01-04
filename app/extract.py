from __future__ import annotations

import logging
from argparse import ArgumentParser

import numpy as np
from implicit.approximate_als import AlternatingLeastSquares
from implicit.datasets.million_song_dataset import get_msd_taste_profile
from implicit.nearest_neighbours import bm25_weight


def extract(output_file: str):
    model = AlternatingLeastSquares(factors=64, dtype=np.float32)
    tracks, users, plays = get_msd_taste_profile()

    logging.info("Weighting matrix by bm25_weight")
    plays = bm25_weight(plays, K1=100, B=0.8)

    logging.info("Compressing matrix")
    plays = plays.tocsr()

    logging.info("Training model")
    model.fit(plays)

    logging.info("Saving item factors")
    factors = model.item_factors
    np.save(output_file, factors)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()
    extract(args.input_file)
