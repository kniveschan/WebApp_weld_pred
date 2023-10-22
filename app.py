from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)


@app.route('/', methods=["get", "post"])  # http://127.0.0.1:5000 + '/' = http://127.0.0.1:5000/
def predict():
    message = ""
    if request.method == "POST":
        with open('models/mor_model_SVR.pkl', 'rb') as mdl:
            saved_model = pickle.load(mdl)

        with open('models/normalize_SVR.pkl', 'rb') as np:
            norm_params = pickle.load(np)

        scaler = MinMaxScaler()
        scaler.min_ = norm_params['min']
        scaler.scale_ = norm_params['scale']

        in_IW = request.form.get("IW")
        in_IF = request.form.get("IF")
        in_VW = request.form.get("VW")
        in_FP = request.form.get("FP")


        input_val = [[float(in_IW), float(in_IF), float(in_VW), float(in_FP)]]

        input_val = scaler.transform(input_val)
        y_pred = saved_model.predict(input_val)

        # print(pred)
        message = f"Глубина шва (depth) {round(y_pred[0][1], 2)} Ширина шва (width) {round(y_pred[0][0], 2)}"


    return render_template("index.html", message=message)


@app.route('/text/')  # http://127.0.0.1:5000 + '/text/' = http://127.0.0.1:5000/text/
def print_text():
    return "<h1>Some text!</h1>"

app.run()