#!/bin/bash

python3 -m pipenv run python3 feature_era_corr_split.py
python3 -m pipenv run python3 era_ft_graph.py
python3 -m pipenv run python3 data_subsets_format.py

python3 -m pipenv run python3 generate_fst_layer_models.py
python3 -m pipenv run python3 model_fst_layer_predict_tournament.py

python3 -m pipenv run python3 extract_snd_layer_training_data.py
python3 -m pipenv run python3 generate_snd_layer_models.py
python3 -m pipenv run python3 model_snd_layer_predict_tournament.py

python3 -m pipenv run python3 final_prediction.py
