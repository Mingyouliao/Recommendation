from flask import Flask, jsonify, request, Response
import pandas as pd
import pickle
import json
import datetime
import numpy as np
import time
from generate_rec import ProductRecommender

app = Flask(__name__)

def load_data():
    NMF_predict = pd.read_csv('./NMF_model.csv', index_col=0)
    SVD_predict = pd.read_csv('./SVD_model.csv', index_col=0)
    SLIM_predict = pd.read_csv('./SLIM_model.csv', index_col=0)
    FISM_predict = pd.read_csv('./FISM_model.csv', index_col=0)
    LLORMA_predict = pd.read_csv('./LLORMA_model.csv', index_col=0)
    
    return NMF_predict, SVD_predict, SLIM_predict, FISM_predict, LLORMA_predict

NMF_predict, SVD_predict, SLIM_predict, FISM_predict, LLORMA_predict = load_data()

#http://127.0.0.1:5000/recommend?user_id=00d0fd357b7a5e18476dec7e46571364

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if user_id:
        recommender = ProductRecommender(user_id, NMF_predict, SVD_predict, SLIM_predict, FISM_predict, LLORMA_predict)
        recommendation_result = recommender.recommend_products()
        #return jsonify(recommendation_result)
        response = json.dumps(recommendation_result, ensure_ascii=False)
        return Response(response, content_type="application/json; charset=utf-8")
    else:
        return jsonify({"error": "No user_id provided"}), 400
    
if __name__ == '__main__':
    app.run(debug=False)