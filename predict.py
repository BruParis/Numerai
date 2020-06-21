import numerox as nx
import os
import numerapi

# download dataset from numerai
data = nx.download('numerai_dataset.zip')

os.system("model_fst_layer_predict_tournament.py")
os.system("model_snd_layer_predict_tournament.py")
os.system("final_prediction.py")
