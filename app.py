import pickle
from flask import Flask, request, app,render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST','GET'])
def predict():
    with open("standardScalar.sav", 'rb') as f:
        scalar = pickle.load(f)
    with open("modelForPrediction.sav", 'rb') as f:
        model = pickle.load(f)
    with open("pca_model.sav", 'rb') as f:
        pca_model = pickle.load(f)

    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]

    data_df = pd.DataFrame(final)
    scaled_data = scalar.transform(data_df)
    principal_data = pca_model.transform(scaled_data)
    predict = model.predict(principal_data)
    if predict[0] == 3:
        result = 'Bad'
    elif predict[0] == 4 :
        result = 'Below Average'
    elif predict[0]==5:
        result = 'Average'
    elif predict[0] == 6:
        result = 'Good'
    elif predict[0] == 7:
        result = 'Very Good'
    else :
        result = 'Excellent'

    
    if result=='Bad':
        return render_template('index.html',pred="The quality for the wine is : " + str(result) + ". This might be the worst thing on this earth. 🤮")
    elif result=='Below Average':
        return render_template('index.html',pred="The quality for the wine is : " + str(result) + ". You should develope good taste.")
    elif result=='Average':
        return render_template('index.html',pred="The quality for the wine is : " + str(result) + ". Try few good drinks.")
    elif result=='Good':
        return render_template('index.html',pred="The quality for the wine is : " + str(result) + ". This is some good stuff, could be better.")
    elif result=='Very Good':
        return render_template('index.html',pred="The quality for the wine is : " + str(result) + ". Now this is the what we call some good taste, Don't worry you can also try few excellent ones.")
    else:
        return render_template('index.html',pred="The quality for the wine is : " + str(result) + ". Best in class 🍷.")


if __name__ == "__main__":
    app.run(debug=True)
