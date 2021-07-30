#from typing_extensions import final
from flask import Flask, render_template, request
import pickle
import csv
import pandas as pd
import numpy as np
#import CatBoost
#import lightgbm
#import joblib
model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


#lgbm_model = pickle.load(open('lgb.pkl', 'rb'))
#XGB_model = pickle.load(open('XGB.pkl', 'rb'))
@app.route('/', methods = ['GET', 'POST'])

def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        f = request.form['csvfile']
        data = []
        with open(f) as file:
            csvfile = csv.reader(file)
            x = pd.DataFrame(csvfile)
            y = x.iloc[1: , 1:]
            my_pred = model.predict(y)
        return render_template('data.html', data = my_pred)





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

#if __name__ == '__main__':
 #   app.run(debug=True)