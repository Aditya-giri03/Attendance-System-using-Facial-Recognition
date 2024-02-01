from flask import Flask, request, render_template
import pickle
import pandas as pd
from flask_cors import CORS, cross_origin
import json

app=Flask(__name__)
CORS(app)

@app.route("/AttendanceApp", methods=["POST","GET"])
def predict():
    data = request.get_json()
    res="";
    
    res={"data":res}
    return json.dumps(res)

@app.route("/cnnArch", methods=["POST","GET"])
def train():
    res=[];
    # algo

    res={"data":res}
    return json.dumps(res);

app.run(debug=True)


#virtualenv venv
#.\\venv\Scripts\activate

#python server.py