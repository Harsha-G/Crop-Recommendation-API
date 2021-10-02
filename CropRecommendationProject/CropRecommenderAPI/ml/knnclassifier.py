import pandas as pd
from .utilityfunc import *

class KnnClassifier:
    def __init__(self):
        artifact_loc = "Artifacts\\"
        artifact_name = "knn_model.pkl"
        artifact = artifact_loc + artifact_name
        self.knn = LoadModelObject(artifact)

        # We will expect our input data to have the same attributes as the elements in col_arr
        self.col_arr = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']      

    def Predict_Class(self, input_data):
        reduced_df = PreProcess_Input(input_data, self.col_arr)
        
        input_arr = reduced_df.values
        predicted_labels = self.knn.predict(input_arr.reshape(1,-1))

        return predicted_labels