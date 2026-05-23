import pandas as pd
import joblib
import streamlit as st
from streamlit_option_menu import option_menu


#print(working_dir)

# loading the saved models
model = joblib.load('opt_svm_model.joblib')
# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Classification System',

                           ['Penguin species Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='species-fill',
                           icons=['animal', 'activity', 'person'],
                           default_index=0)

if selected == 'Penguin species Prediction':

    # page title
    st.title('Penguin Prediction using ML')

    # getting the input data from the user
    island, col2, col3,col4,col5,col6 = st.columns(6)

    with island:
        island = st.selectbox("Island:", ["Biscoe", "Dream", "Torgersen"])

    with col2:
        bill_length_mm = st.number_input("Enter bill length", format="%.1f", value=30.0, min_value=30.0, max_value=60.0,step=0.1)

    with col3:
        bill_depth_mm = st.number_input("Enter bill depth", format="%.1f", value=12.0, min_value=12.0, max_value=20.0,step=0.1)

    with col4:
        flipper_length_mm = st.number_input("Enter flipper length", format="%d", value=150, min_value=150, max_value=300,
                                        step=1)
    with col5:
        body_mass = st.number_input("Enter body mass", format="%d", value=3000, min_value=3000, max_value=6000,
                                        step=1)
    with col6:
        sex = st.selectbox("Gender:", ["Male", "Female"])

    # code for Prediction
    spec_classification = ''

    keys = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]

    # Combine them into a dictionary

    # creating a button for Prediction
    #user_input = [island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass,
                  #gender]
    #input_dict = dict(zip(keys, user_input))
    #st.write(input_dict)

    if st.button('Penguin Prediction Result'):
        user_input = [island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass,
                     sex]
        input_dict = dict(zip(keys, user_input))
        spec_classification = model.predict(pd.DataFrame(input_dict,index=[0])).tolist()[0]
    
    st.success(spec_classification)
