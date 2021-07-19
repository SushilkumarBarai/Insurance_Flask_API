# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:24:13 2021

@author: Sushilkumar
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import logging 
logging.basicConfig(filename='error.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

app = Flask(__name__)
modelH = pickle.load(open('health_insc_model.pkl', 'rb'))


@app.route('/home')
def home():
    return jsonify({ "name": "sushil"})

@app.route('/predictHealthClaim',methods=['GET','POST'])
def predictClaim():
    if request.method=="POST":
        req_Json=request.json
        age=req_Json["age"]
        sex=req_Json["sex"]
        bmi=req_Json["bmi"]
        children=req_Json["children"]
        smoker=req_Json["smoker"]
        region=req_Json["region"]
        charges=req_Json["charges"]
        app.logger.debug(req_Json)
        listkeys=[k for k in req_Json.keys()]
        listvalues = [v for v in req_Json.values()]
        # Printing key list and value in list seprately
        app.logger.debug(listkeys)
        app.logger.debug(listvalues)
        results=modelH.predict([listvalues])
        final=results[0]
        app.logger.debug(final)
        return jsonify({ "claim": int(final)}), 200, {'ContentType':'application/json'} 
    elif request.method=="GET":
        app.logger.debug("get called")
        return jsonify({ "Error": "GET Method Not allowed Call POST METHOD"}),405
        
    

if __name__ == "__main__":
    app.run(debug=False)