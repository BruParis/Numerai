#!/bin/bash

pip install --no-cache-dir tensorflow

pipenv run python feature_era_corr_split.py
pipenv run python era_ft_graph.py
pipenv run python data_subsets_format.py
pipenv run python generate_models.py
pipenv run python model_predict_tournament.py