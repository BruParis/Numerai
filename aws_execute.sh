#!/bin/bash

python3 -m pipenv run python feature_era_corr_split.py
python3 -m pipenv run python era_ft_graph.py
python3 -m pipenv run python data_subsets_format.py
python3 -m pipenv run python generate_fst_layer_models.py
python3 -m pipenv run python model_fst_layer_predict_tournament.py
