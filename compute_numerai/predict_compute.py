# import numerox as nx
import os
import errno
import numerapi

print("START predict.py")

# download dataset from numerai
# nx.download('numerai_dataset.zip')

napi = numerapi.NumerAPI(verbosity="info")
napi.download_current_dataset(unzip=False, dest_filename='current_dataset.zip')

print("DOWNLOAD FINISHED")

try:
    os.makedirs('data')
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Error with : make dir data")
        exit(1)

os.system("unzip -p current_dataset.zip numerai_tournament_data.csv > data/numerai_tournament_data.csv")

os.system("rm current_dataset.zip")

os.system("python src/main.py fst cluster prediction")

os.system("python src/main.py snd cluster prediction")

os.system("python src/main.py snd cluster final_prediction")

os.system("python src/main.py full cluster valid")

os.system("python src/main.py snd cluster upload")

os.system("rm data/*.csv")
os.system("rm data_clusters/pred*.csv")
os.system("rm data_clusters/final*.csv")
os.system("rm data_clusters/snd_layer/*.csv")
