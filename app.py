import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.write(
    """
    # Simple Iris Flower Prediction

    This app predict the iris flower type
    """
)

st.sidebar.header('User Input Parameter')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 0.1, 10.0, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 0.1, 10.0, 4.5)
    petal_length = st.sidebar.slider('Petal Length', 0.1, 10.0, 6.5)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 10.0, 5.4)
    data = {
        'SepalLengthCm' : sepal_length,
        'SepalWidthCm' : sepal_width,
        'PetalLengthCm' : petal_length,
        'PetalWidthCm' : petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

model = pickle.load(open('model/model_softmax.pkl', 'rb'))
prediction = model.predict(df)
predict_proba = model.predict_proba(df)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(predict_proba)