from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('gold_price_prediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    spx = float(request.form['SPX'])
    uso = float(request.form['USO'])
    slv = float(request.form['SLV'])
    euro = float(request.form['EUR/USD'])

    input_data = pd.DataFrame([[spx, uso, slv, euro]],
                              columns=['SPX', 'USO', 'SLV', 'EUR/USD'])
    
    prediction = model.predict(input_data)[0]
    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
