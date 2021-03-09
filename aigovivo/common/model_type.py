import enum


class ModelType(enum.Enum):
    UnivPolyInterpo = 0
    RandomForest = 1
    XGBoost = 2
    K_NN = 3
    NeuralNetwork = 4