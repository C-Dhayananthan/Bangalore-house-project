from flask import Flask,render_template, request
import os
import joblib as jb
import numpy as np
import pandas as pd
import cols


app = Flask(__name__)

@app.route("/",methods = ['GET'])
def home():
    return render_template("templates/index.html")

@app.route("/predict",methods= ["POST"])
def predict():
    if request.method == 'POST':
        size = float(request.form.get("size"))
        tqft = float(request.form.get("tsqft"))
        bath = float(request.form.get("bath"))
        location = str(request.form.get("location"))
        model = jb.load("model2")
        print(location)
        print(type(location))
        ## price predict
        def predict_price(location,sqft,bath,bhk):
            loc_index = np.where(cols.columns2()==location)[0][0]
            x = np.zeros(len(cols.columns2()))
            x[0] = bhk
            x[1] = sqft
            x[2] = bath
            if loc_index >= 0:
                x[loc_index] = 1
                df = pd.DataFrame(x,index=cols.columns2()).transpose()
                return round(np.expm1(model.predict(df).item()),3)
        output = predict_price(location,size,tqft,bath)
        return render_template("templates/index.html" , predicted = output)
    else:
        return render_template("templates/index.html")

if __name__ == "__main__" :
    app.run(debug = True)
