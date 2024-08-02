import pickle
import json
import pandas as pd
import numpy as np
import datetime
import time

def timeit(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        print(f"{method.__name__} time: {end_time - start_time} second")
        return result
    return timed

def recommend_products(user_id, r_hat_predict, recommend_products_num):
    # Get the user's rating predictions
    user_ratings = r_hat_predict.loc[user_id, :]
    
    # Find the indices of the top N ratings
    top_n_indices = np.argsort(-user_ratings.values)[:recommend_products_num]
    
    # Get the top N ratings using the indices
    top_n_ratings = user_ratings.iloc[top_n_indices]
    
    top_n_ratings = [(name, float(score), name) for name, score in top_n_ratings.items()]

    return {"推薦的商品名稱": top_n_ratings}

def save_recommendations_to_csv(recommendations, group, user_id, file_name="recommendations.csv"):
    rows = []
    for rec_type in recommendations["RecommendationTypes"]:
        model_name = rec_type["model"]
        row = {"group": group, "user": user_id, "model": model_name}
        for idx, item in enumerate(rec_type["items"], start=1):
            row[f"第{idx}名"] = item["name"]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(file_name, index=False)
    print(f"Recommendations saved to {file_name}")

class ProductRecommender:
    @timeit
    def __init__(self, user_id, group, NMF_predict, SVD_predict, SLIM_predict, FISM_predict, LLORMA_predict):
        self.user_id = user_id
        self.group = group
        self.recommend_products_num = 10
        self.NMF_predict = NMF_predict
        self.SVD_predict = SVD_predict
        self.SLIM_predict = SLIM_predict
        self.FISM_predict = FISM_predict
        self.LLORMA_predict = LLORMA_predict
        
    
    @timeit
    def NMF_model(self):
        return recommend_products(self.user_id, self.NMF_predict, self.recommend_products_num)
    
    @timeit
    def SVD_model(self):
        return recommend_products(self.user_id, self.SVD_predict, self.recommend_products_num)
    
    @timeit
    def SLIM_model(self):
        return recommend_products(self.user_id, self.SLIM_predict, self.recommend_products_num)
    
    @timeit
    def FISM_model(self):
        return recommend_products(self.user_id, self.FISM_predict, self.recommend_products_num)
    
    @timeit
    def LLORMA_model(self):
        return recommend_products(self.user_id, self.LLORMA_predict, self.recommend_products_num)
    

    
    @timeit
    def recommend_products(self):
        
        recommendations_temp = {
            "NMF_model" : self.NMF_model(),
            "SVD_model" : self.SVD_model(),
            "SLIM_model" : self.SLIM_model(),
            "FISM_model" : self.FISM_model(),
            "LLORMA_model" : self.LLORMA_model(),
        }
        
        recommendations = {"RecommendationTypes": []}

        for model_name, model_data in recommendations_temp.items():
            model_recommendations = {"model": model_name, "items": []}
            
            for item in model_data.get("推薦的商品名稱", []):
                item_dict = {"name": item[0]}
                if len(item) == 2:
                    _, ID = item
                    item_dict["ID"] = ID
                elif len(item) == 3:
                    ID1, middle_item, third_item = item
                    try:
                        middle_item = float(middle_item)
                        item_dict["score"] = middle_item
                        item_dict["ID"] = third_item
                    except ValueError:
                        item_dict["item_number"] = middle_item
                        item_dict["price"] = third_item
                
                model_recommendations["items"].append(item_dict)
                
            recommendations["RecommendationTypes"].append(model_recommendations)
        return recommendations


if __name__ == "__main__":
    # user_id_example = '00d0fd357b7a5e18476dec7e46571364'
    user_groups = {
            "group1": ["bb1eb9a1ee6d45bccb525e8cc60c0d82", "cb637b3a23f4f2703b35d1f36f1e4832", "28959d2742dc2692f1597c91b38d595b"],
            "group2": ["445acd9ddb9eeec60c63c24b01f8abfd", "5f802fcfb96129c5809a0e9544bdd7a0", "db09131797e2d30ecf6e878689af7214"],
            "group3": ["00280f1e800e9ade3424588719fd4c85", "c6690c34f36bc6e3731ae38b3d7810e4", "f73d8df78eab9273feb2974e63fb4dd6"],
            "group4": ["b8684acfcc50f6438a66f32df1a2b7ad", "0f0a4d67d2497c720ff931e95b11e4c7", "616ff1a41b79efa48508b8841736b3f9"],
            "group5": ["a8d96d690a18ec8fd9c81f3fa27b4cc3", "bac1451ec85a8d9ac4febfe24c475404", "01fd4e54996c2b78dd17e3e077bcacf1"],
            "group6": ["ed80ce9e431e9d91e17a758584d1a03a", "d20f4d94797b8d5b2131ec307ef469f5", "ce170c08b9042849dfaecff98926771e"],
            "group7": ["956589c7351d792548083ecd1b09a99d", "ee49b5fd852722d69b934ff8e65d0a0c", "6f99276dcc7ca8b8274975bbf6d1495d"]

    }
    @timeit
    def load_data():
        NMF_predict = pd.read_csv('./NMF_model.csv', index_col=0)
        SVD_predict = pd.read_csv('./SVD_model.csv', index_col=0)
        SLIM_predict = pd.read_csv('./SLIM_model.csv', index_col=0)
        FISM_predict = pd.read_csv('./FISM_model.csv', index_col=0)
        LLORMA_predict = pd.read_csv('./LLORMA_model.csv', index_col=0)
        
        return NMF_predict, SVD_predict, SLIM_predict, FISM_predict, LLORMA_predict

    NMF_predict, SVD_predict, SLIM_predict, FISM_predict, LLORMA_predict = load_data()

    
    # recommender = ProductRecommender(user_id_example, NMF_predict, SVD_predict, SLIM_predict, FISM_predict, LLORMA_predict)
    # recommendation_result = recommender.recommend_products()
    # print(recommendation_result)
    # recommender.save_recommendations_to_csv(recommendation_result)
    
    all_recommendations = []
    for group, users in user_groups.items():
        for user_id in users:
            recommender = ProductRecommender(user_id, group, NMF_predict, SVD_predict, SLIM_predict, FISM_predict, LLORMA_predict)
            recommendation_result = recommender.recommend_products()
            print(group)
            print(user_id)
            all_recommendations.append((group, user_id, recommendation_result))

    for group, user_id, recommendation_result in all_recommendations:
        save_recommendations_to_csv(recommendation_result, group, user_id, file_name=f"recommendations_{group}_{user_id}.csv")