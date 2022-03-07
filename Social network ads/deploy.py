from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,3)
    print(final_features)
    prediction = model.predict(final_features)
    L_collection = {0:"No purchased", 1:"Purchased"}
    result=L_collection[prediction[0]]
    print(result)
    return render_template('result.html', prediction_text="YOU HAVE  {}".format(result))

if __name__=='__main__':
    app.run(port=5000)

