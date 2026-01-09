import streamlit as st
import  pandas as pd

email = st.text_input('Enter email')
password = st.text_input('Enter password')
gender = st.selectbox('Select gender',['male','female'])

btn = st.button('Login karo')

if btn:
    if email == 'gondaliyadev007@gmail.com' and password == '12345678':
        st.balloons()
        st.write(gender)
    else:
        st.error('Login failed')

file = st.file_uploader('Upload a csv file')

if file is not None:
    df = pd.read_csv(file)
    st.dataframe(df.describe())
