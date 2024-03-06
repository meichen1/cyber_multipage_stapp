
# import sys
# sys.path.insert(0, '../')

import dill
import streamlit as st
import io
import os
import pandas as pd
import altair as alt
import plotly.express as px
import time
from openai import OpenAI
from src.helper import cutCatTranformer, columnDropperTransformer, AttackTypeMapping, Prediction_Report
from sklearn.model_selection import train_test_split
from Home import client
from Home import load_data


st.set_page_config(
page_title="Cyber Attack Prediction",
layout="wide")

pd.set_option("styler.render.max_elements", 2000000)





if "OPENAI_API_KEY" not in st.session_state:
    st.write('Your OpenAI API key has not been saved yet! Please save your OpenAI API key in the sidebar.')
    st.session_state["OPENAI_API_KEY"] = ''
    
openai_key = st.sidebar.text_input("Your OpenAI API Key",value=st.session_state['OPENAI_API_KEY'], key=None, type='default')
saved = st.sidebar.button("Save")
if saved:
    st.session_state['OPENAI_API_KEY'] = openai_key
    # st.write('Your OpenAI API key has been saved!')
    
    


try:
    with open('trained_model/xgb_type_pipeline.dill', 'rb') as f:
        model = dill.load(f)
except dill.UnpicklingError:
    print("File 'yourfile.dill' is not a pickled file or it is corrupted.")

#@st.cache_data(experimental_allow_widgets=True) 



if not st.session_state['file']:
    st.write('Please upload your data file, no session state file found')
    
else:
    try:
        print(st.session_state['file'])
        df = load_data(st.session_state['file'])
    except Exception as e:
        st.write(str(e))

    if df is not None:
        X,y = df.drop(columns=['Attack_label', 'Attack_type']), df['Attack_type']
        
        st.write('❤️ You have successfully uploaded your data!')
        st.subheader('Data')
        st.write(df.head())
        st.write('The number of rows:', df.shape[0], ', The number of columns:', df.shape[1])

        # Add a new column called 'select' with default value False as the first column
        # X.insert(0, 'select', False)            
        # edited_df = st.data_editor(
        #                 X,
        #                 column_config={
        #                     "select": st.column_config.CheckboxColumn(
        #                         "select",
        #                         help="Select your **network traffic** for attack prediction",
        #                         default=False,
        #                     )
        #                 },
        #                 hide_index=True
        #             )           
        
        
        # set a slider for sample data for model prediction
        st.subheader('Detection')
        values = st.slider(
            '⚡ Select the percentage of records you want for scanning',
            0.0, 100.0, (20.0))
        st.write('I\'ll scan', values,'%', 'of the records for you.')



# customized styling func
def highlight_attack(val):
    color = 'white' if val == 'Normal' else '#ffe6e6'
    return f'background-color: {color}'


# if successfully read in the dataset -> display the button    
if st.button('Scan your records'):
    try:
        X_train, X, y_train, y = train_test_split(X, y, test_size=values/100.0)

        # set the selected rows for prediction
        # selected_indices = edited_df.loc[edited_df["select"] == True].index.values
        # X = X.loc[selected_indices]
        # y = y.loc[selected_indices]


        # call labelEncoder_y.encode in the helper.py to encode the 15 multiclasses
        attackTypeMapping = AttackTypeMapping()
        y = attackTypeMapping.map_type2value(y)

        # predict
        predictions = model.predict(X)
        
        # progress bar
        progress_text = "In progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()

        # print the prediction
        st.subheader('Detection Results')
        st.write('👨🏿‍💻 Here are the results! Suspicious cyber attack records were highlighted in red!')
        prediction_output = pd.DataFrame(attackTypeMapping.map_value2type(predictions), columns = ['Attack_Type_Prediction'])
        prediction_output = pd.concat([prediction_output, X.reset_index(drop=True)], axis=1)
        st.dataframe(prediction_output.style.applymap(highlight_attack, subset=['Attack_Type_Prediction']))

        # report
        st.subheader('Detection Report')
        st.write('👨🏿‍💻 Here are the prediction report of all types of cyber attack!')
        prediction_report = Prediction_Report()

        precision, recall, fscore = prediction_report.report_precision_recall(y, predictions)
        col1, col2, col3 = st.columns(3)
        col1.metric("precision", precision)
        col2.metric("recall", recall)
        col3.metric("f score", fscore)
        
        

        col1, col2 = st.columns(2)
        fig = prediction_report.plot_confusion_matrix(y, predictions, attackTypeMapping)
        col1.subheader('Confusion Matrix')
        col1.pyplot(fig)

        # check feature importance
        col2.subheader('Check Important Features')
        col2.write('⭐ Here are the feature importance on the default dataset generated from the Permutation Importance.')
        col2.write('⭐ Consider monitoring these features for cyber attack!')
        col2.image('./images/Feature Importance.png')
        
        st.subheader('🤖 Hi! You can consult with your AI assistant for more information!')
        
    except Exception as e:
        st.write(str(e))
        
    
        



        

# # Set OpenAI API key from Streamlit secrets
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Chat with me!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})        
        
    

