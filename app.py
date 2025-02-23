from flask import Flask,request, render_template
import numpy as np
import joblib
import sklearn
print(sklearn.__version__)
# loading models
hybrid_model = joblib.load("Hybrid_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item  = request.form['Item']

        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = hybrid_model.predict(transformed_features).reshape(1,-1)

        return render_template('index.html',prediction = prediction[0])

if __name__=="__main__":
    app.run(debug=True)