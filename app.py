from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, static_url_path='/static')

model = pickle.load(open('models/ConstructionCostEstimator.pkl', 'rb'))

location_map = {'rural': 0, 'urban': 1, 'suburban': 2, 'metropolitan': 3}
quality_map = {'low': 0, 'medium': 1, 'high': 2}
furnished_map = {'Non-Furnished': 0, 'Furnished': 1}
luxury_map = {'basic': 0, 'standard': 1, 'luxury': 2}
type_map = {'commercial': 0, 'residential': 1}


@app.route('/')
def home():
    return render_template("index.html", prediction_text=None)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    type = request.form['type']
    floor = int(request.form['floor'])
    sqft = float(request.form['sqft'])
    quality = request.form['quality']
    furnished = request.form['furnished']
    luxury = request.form['luxury']

    location_encoded = location_map.get(location, -1)
    quality_encoded = quality_map.get(quality, -1)
    furnished_encoded = furnished_map.get(furnished, -1)
    luxury_encoded = luxury_map.get(luxury, -1)
    type_encoded = type_map.get(type, -1)

    features = np.array(
        [[location_encoded, type_encoded, floor, sqft, quality_encoded, furnished_encoded, luxury_encoded]])

    prediction = model.predict(features)

    predicted_price = "₹{:,.0f}".format(prediction[0])

    return render_template('index.html', prediction_text=predicted_price)


if __name__ == "__main__":
    app.run(debug=True)
