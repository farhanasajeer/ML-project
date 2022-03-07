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
    #request.value will help us to read the particular value in the form
    exp= float(request.values['experience'])
    #convert the data into the two dimensional data
    exp=np.reshape(exp,(-1,1))
    output=model.predict(exp)
    #select the particular item
    output=output.item()
    #rounding the output value to two decimal point
    output=round(output,2)
    return render_template('result.html', prediction_text="congrats!!...you are eligble for a salary of Rs.{}".format(output))
if __name__=='__main__':
    app.run(port=5000)
    