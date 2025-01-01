import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
import plotly.express as px 
import joblib

# Add header and Subheader 
st.markdown("<h1 style = 'color: #2A004E; text-align: center; font-size: 40px; font-family: helvetica'>Laptop Price Prediction Web App</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #5CB338; text-align: center; font-family: helvetica '>Built By Dhanny123 </h4>", unsafe_allow_html = True)

# Image 
col1, col2, col3 = st.columns([1, 2, 1])
col2.image('lappy.png' )

# Background To project 
st.markdown("<h4 style = 'margin: -30px; color: #0E2954; text-align: center; font-family: helvetica '>Background Of The Project</h4>", unsafe_allow_html = True)
st.write("""
 In the rapidly evolving world of technology, laptops have become indispensable tools for professionals, students, and everyday users alike. With the sheer variety of options available, understanding the factors that influence laptop pricing can be a daunting task. This challenge inspired the creation of a machine learning model designed to predict laptop prices with remarkable accuracy.
         
In 2023, a team of data scientists observed a growing demand for tools to assist consumers in making informed purchasing decisions. Despite the wealth of online reviews and comparison platforms, the lack of transparent and reliable price prediction models posed a significant challenge. The complexity stemmed from the numerous variables involved, including hardware specifications, brand reputation, market trends, and even geographical location.
Specifications such as processor type, ram, storage, and display quality. brand and model details were also considered.

The laptop price prediction model has proven to be a game-changer for both consumers and retailers. Key applications include:

Empowering users with price transparency and helping them identify the best value for their budget.
Enabling dynamic pricing strategies based on market trends and customer preferences.

As the model continues to evolve, it serves as a testament to the transformative power of data-driven solutions in modern commerce.

Conclusion
         
The journey of creating the laptop price prediction model underscores the importance of innovation and collaboration. By combining cutting-edge technology with real-world data, the team has made a lasting impact on the way people approach laptop purchases, setting a new standard for transparency and efficiency in the tech industry.""")

st.divider()

st.markdown("<br>", unsafe_allow_html= True)
data = pd.read_csv('laptop_price_clean.csv')
st.dataframe(data, use_container_width = True)


st.sidebar.image('user.png', caption= 'Welcome!!!')

# Feature Input 
brand = st.sidebar.selectbox('Laptop brand', data.brand.unique(), index=1)
color = st.sidebar.selectbox('Laptop Color', data.color.unique(), index=0)
graphics = st.sidebar.selectbox('Laptop graphics', data.graphics.unique(), index=0)
harddisk = st.sidebar.number_input('Laptop harddisk Size (GB)', min_value = 0.0,max_value = 10000.0, value = data.harddisk.median())
model = st.sidebar.selectbox('Laptop model', data.loc[data.brand == brand]["model"].unique())
ram = st.sidebar.number_input('Laptop ram Size (GB)', min_value =0.0, max_value =5000.0, value = data.ram.median())
opsys = st.sidebar.selectbox('Laptop OS', data.loc[data.brand == brand]["OS"].unique())
cpu = st.sidebar.selectbox('Laptop CPU', data.loc[data.brand == brand]["cpu"].unique())
scrSize = st.sidebar.number_input('Laptop Screen Size (Inches)', min_value =0.0, max_value =500.0, value = data['screen_size'].median())


inputs = {
    'brand': [brand],
    'model': [model],
    'screen_size': [scrSize],
    'color': [color],
    'harddisk': [harddisk],
    'cpu': [cpu],
    'ram': [ram],
    'OS': [opsys],
    'graphics': [graphics],
}    


inputVar = pd.DataFrame(inputs)

st.divider()
st.header('User Input')
st.dataframe(inputVar)


# Inport the transformers 
brand_scaler = joblib.load('brand_encoder.pkl')
color_scaler = joblib.load('color_encoder.pkl')
cpu_scaler = joblib.load('cpu_encoder.pkl')
graphics_scaler = joblib.load('graphics_encoder.pkl')
harddisk_scaler = joblib.load('harddisk_scaler.pkl')
model_scaler = joblib.load('model_encoder.pkl')
opsys_scaler = joblib.load('OS_encoder.pkl')
ram_scaler = joblib.load('ram_scaler.pkl')
scrSize_scaler = joblib.load('screen_size_scaler.pkl')

# Use the imported transformers to transform the users input
inputVar['brand'] = brand_scaler.transform(inputVar[['brand']])
inputVar['color'] = color_scaler.transform(inputVar[['color']])
inputVar['cpu'] = cpu_scaler.transform(inputVar[['cpu']])
inputVar['graphics'] = graphics_scaler.transform(inputVar[['graphics']])
inputVar['harddisk'] = harddisk_scaler.transform(inputVar[['harddisk']])
inputVar['model'] = model_scaler.transform(inputVar[['model']])
inputVar['OS'] = opsys_scaler.transform(inputVar[['OS']])
inputVar['ram'] = ram_scaler.transform(inputVar[['ram']])
inputVar['screen_size'] = scrSize_scaler.transform(inputVar[['screen_size']])


pricemodel = joblib.load('Laptopmodel.pkl')

predictButton = st.button('Click To Predict')

if predictButton:
    predicted = pricemodel.predict(inputVar)
    st.success(f'The predicted Laptop Price is ${round(predicted[0], 2)}')



