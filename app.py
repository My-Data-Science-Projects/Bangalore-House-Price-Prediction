import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

pickle_in = open('rfr.pkl','rb')
regressor = pickle.load(pickle_in)

df = pd.read_csv('clean_data.csv')

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)


def predict_price(model, bath, balcony, total_sqft_int, bhk, price_per_sqft):
    x =np.zeros(len(X.columns)) # create zero numpy array, len = 107 as input value for model

    # adding feature's value accorind to their column index
    x[0]=bath
    x[1]=balcony
    x[2]=total_sqft_int
    x[3]=bhk
    x[4]=price_per_sqft

    #print(x)

    # feature scaling
    # x = sc.transform([x])[0] # give 2d np array for feature scaling and get 1d scaled np array
    #print(x)

    return model.predict([x])[0]


def main():
    st.title("Bangalore House price Prediction")
    html_temp = """
    <h2 style="color:white;text-align:left;"> Streamlit House prediction ML App </h2>
    """

    st.markdown(html_temp,unsafe_allow_html=True)
    st.subheader('Please enter the required details:')
    bath = st.text_input("No. of Bathroom","")
    balcony = st.text_input("No. of Balcony","")
    total_sqft_int = st.text_input("Total Sqft","")
    bhk = st.text_input("No. of BHK","")
    price_per_sqft = st.text_input("Price Per Sqft","")

    result=""

    if st.button("Predict Price in Lakhs"):
        result=predict_price(regressor, bath, balcony, total_sqft_int, bhk, price_per_sqft)
    st.success('The output is {}'.format(result*100000))


if __name__=='__main__':
    main()
        
