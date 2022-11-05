import numpy as np
# from xgboost import XGBClassifier
# from sklearn.ensemble import XGBoost as xgb
# from xgboost.sklearn import XGBClassifier
import pickle
# from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
import pandas as pd
import csv
from flask import Flask, request, render_template



# import joblib


app = Flask(__name__)
model = pickle.load(open('model_new.pkl', 'rb'))


# model = XGBClassifier.Booster({'nthread':4})
# model.load_model('new.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    #with open('newfile.csv',newline='') as csv_file:
     #   data = csv.DictReader(csv_file)

      #  for col in data:
#  #   print(col('radius'))
     #       data = [float(x) for x in request.form.values()]

    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
#        feature_vls = [np.array(data)]


    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                     'mean smoothness', 'mean compactness', 'mean concavity',
                     'mean concave points', 'mean symmetry', 'mean fractal dimension',
                     'radius error', 'texture error', 'perimeter error', 'area error',
                     'smoothness error', 'compactness error', 'concavity error',
                     'concave points error', 'symmetry error', 'fractal dimension error',
                     'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                     'worst smoothness', 'worst compactness', 'worst concavity',
                     'worst concave points', 'worst symmetry', 'worst fractal dimension']

    df = pd.DataFrame(features_value, columns=features_name)
#    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 0:
        res_val = "** Cancer Detected **"
    else:
        res_val = "No Cancer Detected"

    return render_template('index.html', prediction_text='In Patient data {}'.format(res_val))

if __name__ == "__main__":
    #     app.debug = True
    app.run()



