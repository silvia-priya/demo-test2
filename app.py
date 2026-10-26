from flask import Flask,request,jsonify
import joblib,numpy as np

app=Flask(__name__)

model=joblib.load("filename.pkl")

@app.route("/",methods=["GET"])
def home():
    return "Welcome to prediction usin CICD"

@app.route("/predict",methods=["POST"])
def predcit():
    try:
        data=request.get_json()

        features=np.data["features"].reshape(1,-1)
        predicted=model.predict(features)[0]
        classes=["setosa","versicolor","virginica"]
        result=classes[predicted]
        return jsonify(result)

    except Exception as e:
        return jsonify({"Error":e}),400


if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000) #comment1