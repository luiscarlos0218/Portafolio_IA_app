#  We ensure proper path handling in Python
import Definitions
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.ModelController import ModelController

### Setup and configuration

st.set_page_config(
    layout="centered", page_title="Image Classifier", page_icon="❄️"
)

### My vars

@st.cache_resource
def load_controller():
    return ModelController()

ctrl = load_controller()

### My UI starting here

with st.form(key="my_form"):

    uploaded_file = st.file_uploader(
        "Choose a CSV file", accept_multiple_files=False, type="csv"
    )

    submit_button = st.form_submit_button(label="Submit")

if submit_button and uploaded_file is not None:
    input_df, is_valid = ctrl.load_input_data(uploaded_file)
    st.session_state["input_df"] = input_df if is_valid else None

input_df = st.session_state.get("input_df")

if input_df is not None:
    st.caption("✅ This is your data")
    event = st.dataframe(
        input_df,
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
    )
    st.caption("▶ Please select a row")

    if event is not None and event.selection.rows:        
        current_row_index = event.selection.rows[0]
        current_row = input_df.iloc[current_row_index]

        #TO-DO: Llama la clase de predicción para procesar la información
        X, Y, Y_pred = ctrl.predict(current_row)
        #TO-DO: Obten el nombre de las clases
        class_names = ctrl.get_categories() 

        real_letter = class_names[int(Y)]
        pred_letter = class_names[int(Y_pred[0])]


        col1, col2 = st.columns([1, 2])  

        with col1:
            st.caption("🗣 Your Prediction")
            st.write(pred_letter)



        with col2:
            st.caption("🎯 Your results")
            st.metric("Real", real_letter)
            st.metric("Prediction", pred_letter)
