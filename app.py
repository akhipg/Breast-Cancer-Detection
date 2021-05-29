from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("breastcancer.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    final=[np.array(float_features)]
    print(float_features)
    print(final)
    prediction=model.predict(final)
    #output='{0:.{1}f}'.format(prediction[0][1], 2)

    if(prediction[0]==0):
        return render_template('breastcancer.html',pred='The Breast Cancer is Malignant')
    else:
        return render_template('breastcancer.html',pred='The Breast Cancer is Benign')


if __name__ == '__main__':
    app.run(debug=True)