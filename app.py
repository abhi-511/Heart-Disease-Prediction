from flask import Flask , render_template , request
import numpy as np
import pickle
import pandas as pd




app = Flask(__name__)
model = pickle.load(open('hdp_model.pkl', 'rb'))

@app.route("/",)
def hello():
    return render_template("index.html")


@app.route("/sub", methods = ["POST"])
def submit():
    # Html to py
    if request.method == "POST":
        name = request.form["Username"]

    return render_template("sub.html", n = name)


@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
        name = request.form["Username"]

    input_features = [float(x) for x in request.form.value()]
    features_value = [np.array(input_features)]

    features_names = ["age", "sex", "cp", "trestbps", "chol", "fbd", "restecg", "thalach", "excang", "oldpeak", "slope", "ca", "thal"]

    df = pd.DataFrame(features_value, columns=features_names)
    output = model.predict(df)

    if output == 0:
        res_val = " Heart Disease"
    else:
        res_val = "no Heart Disease"


        return render_template('predict.html', prediction_text = 'The Patient has {}'.format(res_val), n = name)



if __name__=="__main__":
    app.run(debug=True)