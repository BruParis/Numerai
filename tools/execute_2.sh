python src/main.py fst cluster set_h5
python src/main.py fst cluster ft_era_corr split_data train prediction
python src/main.py snd cluster split_data train prediction
python src/main.py snd cluster final_prediction
