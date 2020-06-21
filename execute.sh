#!/bin/bash

# pipenv run python3 feature_era_corr_split.py
# pipenv run python3 era_ft_graph.py
# pipenv run python3 data_subsets_format.py

# pipenv run python3 generate_full_models.py
# pipenv run python3 model_fst_layer_predict_tournament.py -f

# pipenv run python3 generate_fst_layer_models.py
pipenv run python3 model_fst_layer_predict_tournament.py

# pipenv run python3 extract_snd_layer_training_data.py
# pipenv run python3 generate_snd_layer_models.py
pipenv run python3 model_snd_layer_predict_tournament.py

pipenv run python3 final_prediction.py

pipenv run python3 validation_score.py fst
pipenv run python3 validation_score.py snd
pipenv run python3 validation_score.py full
