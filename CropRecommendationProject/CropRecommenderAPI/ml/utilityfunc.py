import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD

def PreProcess_Input(input_data, col_arr):
    input_data = pd.DataFrame(input_data, index=[0])

    # We start out by Normalizing and Standardizing
    preprocessed_df = NormalizeAndStandardize(input_data, col_arr)
    M = preprocessed_df.values

    svd_artifact_loc = "Artifacts\\"
    svd_artifact_name = "cropdata_svdmodel.pkl"

    # Next we reduce our data using Truncated SVD to 3 dimensions
    svd = LoadModelObject(svd_artifact_loc+svd_artifact_name)
    reduced_df = pd.DataFrame(data = svd.transform(M), columns = ['dim1', 'dim2', 'dim3'])
    
    return reduced_df
    
def SaveModelObject(model_name, model_object):
    pickle_out = open(model_name, 'wb')
    pickle.dump(model_object, pickle_out)
    pickle_out.close()
    
def LoadModelObject(model_name):
    try:
        pickle_in = open(model_name, 'rb')
        model = pickle.load(pickle_in)
        return model
    except Exception as e:
        print(e.message)

# Function to perform normaliztion and standardization using preprocessing module from sklearn
# ====================================================================
# Parameters:
# input_df - The input dataframe
# input_col_arr - The input columns we will be interested in
# ====================================================================

def NormalizeAndStandardize(input_df, input_col_arr):
    n_scaler = preprocessing.Normalizer()
    s_scaler = preprocessing.StandardScaler()

    # We expect input_df to have only a single row

    df_n = n_scaler.fit_transform(input_df[input_col_arr])
    df_n = pd.DataFrame(df_n, columns=input_col_arr)    

    standardized_data = s_scaler.fit_transform(df_n.values.reshape(-1,1))     
    df_s = pd.DataFrame(data=standardized_data.reshape(1,-1), columns=input_col_arr)

    return df_s

