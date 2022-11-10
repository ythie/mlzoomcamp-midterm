import pickle
import pandas as pd
import xgboost as xgb

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('solar_irradiation')

@app.route('/predict', methods=['POST'])
def predict():
    datajson = request.get_json()

    data = pd.DataFrame.from_dict(datajson)
    y_pred = model.predict(xgb.DMatrix(data))
    
    result = {'Predicted values' : y_pred}

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)