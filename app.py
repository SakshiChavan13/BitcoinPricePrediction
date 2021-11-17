from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import keras.models
import re
import sys 
import os
import base64
from numpy import array

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(15,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
#model.load_weights('model.h5')

app = Flask(__name__)


def func(date,data):
    flag=0
    for i,j in data.iterrows():
        dataset_date = data.at[i,'Date']
        dt = dataset_date.strftime("%Y-%m-%d")
        
        if (date == dt): 
            flag=1
            len = 15
            price=[]
            while len>0 :
               # print(price)
                i = i-1
               
                len= len-1
                price.append(float(data.at[i,'Price']))
            break


           
    if(flag==1):
        return price
    else:
        return 0
    

   

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])



def predict():
    data = pd.read_csv('data/Bitcoin Historical Data.csv')
    df = pd.to_datetime(data['Date'])
    data['Date'] = df
    data['Price'] = data['Price'].str.replace(',', '') #removing commas from price and converting to float
    data['Price'] = data['Price'].astype(float)
	#sys.modules['sklearn.externals.joblib'] = joblib
	#Alternative Usage of Saved Model
    #lstm_model = load_model('model.h5')

    #lstm_model = open("LSTM_model.pkl","rb")
   # nn = joblib.load(lstm_model)
   # df1=data.reset_index()['Price']
    
    
     
    if request.method == 'POST':
	    
        date = request.form['dte']
        no = request.form['noofday']
        num = int(no)

        price = func(date,data)
        if price==0:
            lst_out ='Date not found in dataset'
            
        else:
            price1 = []
            price1 = func(date,data)
            price1.reverse()
            price2 = pd.DataFrame(price1)
            scaler=MinMaxScaler(feature_range=(0,1))
            price=scaler.fit_transform(np.array(price2).reshape(-1,1))
            x_input=price[:].reshape(1,-1)
            temp_input=list(x_input)
            temp_input=temp_input[0].tolist()
           
            price3 = price.reshape((1, 15 ,1))
            
            lst_output=[]
            n_steps=num
            i=0
            x_input=[]
            while(i<num):
    
                if(len(temp_input)>15):
        
                    x_input=np.array(temp_input[1:])
                    #print("{} day input {}".format(i,x_input))
                    x_input=x_input.reshape(1,-1)
                    x_input=x_input.reshape((1,15,1))
                    #print(x_input)
                    yhat = model.predict(x_input, verbose=0)
                    #print("{} day output {}".format(i,yhat))
                    temp_input.extend(yhat[0].tolist())
                    temp_input=temp_input[1:]
                    #print(temp_input)
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else: #for very first iteration it will enter in this loop
                    x_input=np.array(price3[:])
                    x_input=x_input.reshape(1,-1)
                    
                    x_input = x_input.reshape((1,15,1))
                    yhat = model.predict(x_input, verbose=0)
                    #print(yhat[0])
                    temp_input.extend(yhat[0].tolist())
       
                    lst_output.extend(yhat.tolist())
                    i=i+1
    
           
        
        lst_out=[]
        lst_out = scaler.inverse_transform(lst_output)    
            
             
    return render_template('result.html',prediction = lst_out, dte = date, no =num)
        


if __name__ == '__main__':
	app.run(debug=True)