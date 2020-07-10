# import numerox as nx
import os
import numerapi

print("START predict.py")

# download dataset from numerai
# nx.download('numerai_dataset.zip')

napi = numerapi.NumerAPI(verbosity="info")
napi.download_current_dataset(unzip=False, dest_filename='current_dataset.zip')

print("DOWNLOAD FINISHED")

os.system("unzip -p current_dataset.zip numerai_tournament_data.csv > numerai_tournament_data.csv")

os.system("rm current_dataset.zip")

os.system("python model_fst_layer_predict_tournament.py")

os.system("python model_snd_layer_predict_tournament.py")

os.system("python final_prediction.py")

os.system("python upload_results.py")

os.system("rm *.csv")
os.system("rm data_subsets_036/*.csv")
os.system("rm data_subsets_036/snd_layer/*.csv")
