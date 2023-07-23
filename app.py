import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

st.set_page_config('wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Heart disease Prediction App

This app predicts If a patient has a heart disease

""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe

def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')

    sex  = st.sidebar.selectbox('Sex',(0,1))
    cp = st.sidebar.selectbox('Chest pain type(CP)',(0,1,2,3))
    tres = st.sidebar.number_input('Resting blood pressure(TRES): ')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar(FBS)',(0,1))
    res = st.sidebar.number_input('Resting electrocardiographic results(RES): ')
    tha = st.sidebar.number_input('Maximum heart rate achieved(THA): ')
    exa = st.sidebar.selectbox('Exercise induced angina(EXA): ',(0,1))
    old = st.sidebar.number_input('oldpeak ')
    slope = st.sidebar.number_input('he slope of the peak exercise ST segmen: ')
    ca = st.sidebar.selectbox('number of major vessels(CA)',(0,1,2,3))
    thal = st.sidebar.selectbox('thal',(0,1,2))

    data = {'age': age,
            'sex': sex, 
            'cp': cp,
            'trestbps':tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
heart_dataset = pd.read_csv('dataset.csv')
heart_dataset = heart_dataset.drop(columns=['target'])

df = pd.concat([input_df,heart_dataset],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

df = df[:1] # Selects only the first row (the user input data)

st.write(input_df)


# Reads in saved classification model
# pickle_in= open("model_joblib_heart.pkl","rb")
# rf=pickle.load(pickle_in)
# load_clif = pickle.load(open('model_joblib_heart.pkl','rb'))

# # Apply model to make predictions
# prediction = rf.predict(df)
# # prediction_prob = load_clif.predict_prob(df)


# st.subheader('Prediction')
# st.write(prediction)

# st.subheader('Prediction Probability')
# # st.write(prediction_prob)
model = joblib.load('model_joblib_heart.pkl')

if st.button("Predict"):    
    result=model.predict(input_df)
    if result == 0:
            st.subheader("PERSON IS OUT OF DANGER.")
    else:
            st.write("YES IT SEEMS TO BE A HEART DISEASE")
        