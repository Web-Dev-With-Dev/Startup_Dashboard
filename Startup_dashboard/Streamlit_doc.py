import streamlit as st
import pandas as pd
import time

st.title('Startup Dashboard')
st.header('I am learning Streamlit')
st.subheader('And I am loving it!')

st.write('This is a normal text')
st.markdown("""
### My favorite movies
- Race 3
- Humshakals
- Housefull
""")

st.code("""
def foo(input)
    return foo**2

x = foo(2)
""")

st.latex('x^2 + y^2 + 2 = 0')

df = pd.DataFrame({
    'name': ['Nitish', 'Ankit', 'Anupam'],
    'marks': [50, 60, 70],
    'package': [10, 20, 30]
})

st.dataframe(df)

st.metric('revenue', 'RS 3L', '-3%')

st.json({
    'name': ['Nitish', 'Ankit', 'Anupam'],
    'marks': [50, 60, 70],
    'package': [10, 20, 30]
})

st.image('C:/Users/gonda/OneDrive/Pictures/Screenshots/Screenshot 2025-12-05 085903.png')

st.sidebar.title('About')

col1, col2 = st.columns(2)

with col1:
    st.image('C:/Users/gonda/OneDrive/Pictures/Screenshots/Screenshot 2025-12-05 085903.png')

with col2:
    st.image('C:/Users/gonda/OneDrive/Pictures/Screenshots/Screenshot 2025-12-05 085903.png')

st.error('Login Failed')

st.success('Login Successful')

st.info('Login Successful')

st.warning('Login Successful')

bar = st.progress(0)

for i in range(1, 101):
    time.sleep(0.1)
    bar.progress(i)

email = st.text_input('Enter email')
number = st.number_input('Enter age')
st.date_input('Enter registration date')

