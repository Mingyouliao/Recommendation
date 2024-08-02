from flask import Flask, request, jsonify, Response
import json
import requests

app = Flask(__name__)
#http://127.0.0.1:8080/get_recommendation
# test user : 
# 263f9ae8adc4ed0bbd11419941cb6868
# 0795470fd79361191abdf6a2e5bf8b74
# 3b20d7d5a04031b6bf6f844dfa98423c


def get_recommendation_from_original_api(user_id):
    url = f"http://127.0.0.1:5000/recommend?user_id={user_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

@app.route('/get_recommendation', methods=['GET'])
def get_recommendation():
    user_id = "0795470fd79361191abdf6a2e5bf8b74"
    if user_id:
        data = get_recommendation_from_original_api(user_id)
        response = json.dumps(data, ensure_ascii=False)
        if data:
            return Response(response, content_type="application/json; charset=utf-8")
        else:
            return jsonify({"error": "Unable to get recommends data"}), 404
    else:
        return jsonify({"error": "invalid user_id"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080)

