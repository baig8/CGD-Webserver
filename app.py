import pandas as pd
import csv
from flask import Flask, request, render_template
import numpy as np
import pickle
#from tkinter.filedialog import askopenfilename

from sklearn.ensemble import RandomForestClassifier


# from sklearn.externals import joblib
# import xgboost as xgb
# from xgboost import XGBClassifier
# from sklearn.ensemble import XGBoost as xgb
# from xgboost.sklearn import XGBClassifier

#-----------------------------------------------------

app = Flask(__name__)
#
model = pickle.load(open('model_new.pkl', 'rb'))
#
# #----------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():


    name = request.args.get("name")

    #use filename var which uses tkinter to open a file and pass it as csv

   # print(filename)
    #for only csv ttype files

 #   filetypes =(
  #      ('Text files', '*.csv'),
   # )
   # filename = askopenfilename(title='Select a file...',
    #filetypes=filetypes,)

    with open('newfile.csv', newline='') as csv_file:

#    with open(filename, newline='') as csv_file:
        data = csv.DictReader(csv_file)
        for row in data:
            # print(row)
            # print(row['perimeter'])
            # using key to store the values in a variable and then send to a list

            radius = row['radius']
            texture = row['texture']
            perimeter = row['perimeter']
            area = row['area']
            smoothness = row['smoothness']
            concavity = row['concavity']
            concavePoints = row['concave_points']
            symmetry = row['symmetry']
            fractalDimension = row['fractal_dimension']
            worstRadius = row['worst_radius']
            worstTexture = row['worst_texture']
            worstPerimeter = row['worst_perimeter']
            # print(worstPerimeter)
            worstArea = row['worst_area']
            worstSmoothness = row['worst_smoothness']
            worstCompactness = row['worst_compactness']
            worstConcavity = row['worst_concavity']
            worstConcavePoints = row['worst_concave_points']
            worstSymmetry = row['worst_symmetry']
            worstFractal_dimension = row['worst_fractal_dimension']
            meanRadius = row['mean_radius']
            meanTexture = row['mean_texture']
            meanPerimeter = row['mean_perimeter']
            meanArea = row['mean_area']
            meanSmoothness = row['mean_smoothness']
            meanCompactness = row['mean_compactness']
            meanConcavity = row['mean_concavity']
            meanConcavePoints = row['mean_concave_points']
            meanSymmetry = row['mean_symmetry']

            meanFractalDimension = row['mean_fractal_dimension']
            meanFractalerror = row['mean_fractal_dimension']

            # print(meanFractalDimension)
            auto = [

                radius,
                texture,
                perimeter,
                area,
                smoothness,
                concavity,
                concavePoints,
                symmetry,
                fractalDimension,
                worstRadius,
                worstTexture,
                worstPerimeter,
                # print(worstPerimeter)
                worstArea,
                worstSmoothness,
                worstCompactness,
                worstConcavity,
                worstConcavePoints,
                worstSymmetry,
                worstFractal_dimension,
                meanRadius,
                meanTexture,
                meanPerimeter,
                meanArea,
                meanSmoothness,
                meanCompactness,
                meanConcavity,
                meanConcavePoints,
                meanSymmetry,
                meanFractalDimension,
                meanFractalerror

            ]
            # print(auto)
            # print(auto[0])

#            feature_value = [float (x) for x in request.form.values([np.array(auto)])]
            feature_value = [np.array(auto)]


            # print(feature_value)

            features_name = ['radius',
                             'texture',
                             'perimeter',
                             'area',
                             'smoothness',
                             'concavity',
                             'concavePoints',
                             'symmetry',
                             'fractalDimension',
                             'worstRadius',
                             'worstTexture',
                             'worstPerimeter',
                             # print(worstPerimeter)
                             'worstArea',
                             'worstSmoothness',
                             'worstCompactness',
                             'worstConcavity',
                             'worstConcavePoints',
                             'worstSymmetry',
                             'worstFractal_dimension',
                             'meanRadius',
                             'meanTexture',
                             'meanPerimeter',
                             'meanArea',
                             'meanSmoothness',
                             'meanCompactness',
                             'meanConcavity',
                             'meanConcavePoints',
                             'meanSymmetry',
                             'meanFractalDimension',
                             'mean_fractalerror'
                             ]

            # print(features_name)

            df = pd.DataFrame(feature_value, columns=features_name)
            # print(df)

            output = model.predict(df)

            # THis is the prediction output
            # print(output)

            if output == 0:
                res_val = "---^--^----^-***Cancer Detected***------------"
            else:
                res_val = "---***-No Cancer Detected-***---"

         #   new = res_val

        #     print(res_val)

            return render_template('index.html', name =res_val)
        filename.close()


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
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
    output = model.predict(df)
    if output == 0:
        res_val = "** Cancer Detected **"
    else:
        res_val = "No Cancer Detected"

    return render_template('index.html', prediction_text='In Patient data {}'.format(res_val))







if __name__ == '__main__':
   #home()
   app.run()
