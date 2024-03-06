import streamlit as st 
import dill
import io
import pandas as pd
import altair as alt
import plotly.express as px
import time
from openai import OpenAI
from src.helper import cutCatTranformer, columnDropperTransformer, AttackTypeMapping, Prediction_Report
from sklearn.model_selection import train_test_split
from PIL import Image



st.set_page_config(
page_title="Cyber Attack Detector",
page_icon="🤖",
layout="wide",
initial_sidebar_state="expanded")




if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ''#st.secrets["OPENAI_API_KEY"]

# Add OpenAI API key input in the sidebar
openai_key = st.sidebar.text_input("Your OpenAI API Key",value=st.session_state['OPENAI_API_KEY'], key=None, type='default')
saved = st.sidebar.button("Save")
if saved:
    st.session_state['OPENAI_API_KEY'] = openai_key
    # st.write('Your OpenAI API key has been saved!')


pd.set_option("styler.render.max_elements", 2000000)



# ## load the trained dill model
# model = dill.load(open('trained_model/xgb_type_pipeline.dill', 'rb'))



def switch_page(page_name: str):
    """
    Switch page programmatically in a multipage app

    Args:
        page_name (str): Target page name
    """
    from streamlit.runtime.scriptrunner import RerunData, RerunException
    from streamlit.source_util import get_pages

    def standardize_name(name: str) -> str:
        return name.lower().replace("_", " ")

    page_name = standardize_name(page_name)

    pages = get_pages("Home.py")  # OR whatever your main page is called

    for page_hash, config in pages.items():
        if standardize_name(config["page_name"]) == page_name:
            raise RerunException(
                RerunData(
                    page_script_hash=page_hash,
                    page_name=page_name,
                )
            )

    page_names = [standardize_name(config["page_name"]) for config in pages.values()]
    raise ValueError(f"Could not find page {page_name}. Must be one of {page_names}")


@st.cache_resource
def load_data(file):
    return pd.read_csv(file,low_memory=False)


st.markdown(
"""
<h1 style="text-align: center;">Cyber Attack Detector</h1>
""",
unsafe_allow_html=True,
)

# Load the image
image_path = './images/cyberAttackIllustration.png'
image = Image.open(image_path)

# Calculate the new height based on the desired width-to-height ratio
desired_ratio = 3 / 1
new_width = image.width
new_height = int(new_width / desired_ratio)

# Calculate the cropping coordinates
left = 0
top = (image.height - new_height) // 2
right = image.width
bottom = top + new_height

# Crop the image
cropped_image = image.crop((left, top, right, bottom))
# Display the cropped image
st.image(cropped_image, use_column_width=True)

st.subheader('🤖 Hi! I\'m helping you scan the potential cyberattack from Edge-IIoT network traffic records. ')
st.write('📈 Upload your network traffic records in csv format to start!')


if "file" not in st.session_state:
    st.session_state['file'] = None

# dataset options
option = st.selectbox(
    'What dataset do you want to use?',
    ('I want to use the default dataset', 'I want to upload my own dataset'),
    label_visibility="collapsed",
    index = None,
    placeholder="Select ..."
)

# option = st.radio(
#     'What dataset do you want to use?',
#     ('I want to use the default dataset', 'I want to upload my own dataset'),
#     index=0
# )

def handle_option(option):
    if option == 'I want to use the default dataset':
        st.session_state['file'] = 'datasets/ML-EdgeIIOT-testdata.csv'
    elif option == 'I want to upload my own dataset':
        st.session_state['file'] = st.file_uploader("Upload file", type=["csv"])
    else:
        st.session_state['file'] = None


change = st.button('Change', on_click=handle_option, args=[option,])



if st.session_state['file']:

    try:
        df = load_data(st.session_state['file'])
        X,y = df.drop(columns=['Attack_label', 'Attack_type']), df['Attack_type']

    except Exception as e:
        st.write(str(e))

    # st.write('❤️ You have successfully uploaded your data!')
    st.subheader('Data')
    st.write(df.head())
    st.write('The number of rows:', df.shape[0], ', The number of columns:', df.shape[1])


                
            
    ## insert some space
    st.write("\n")
    st.write("\n")


    st.subheader("Let's explore:")
    st.write("\n")
    col1, col2 = st.columns(2)
    with col1: 
        if st.button("Explore your dataset with our dashboard!"):
            switch_page('Data_Exploration')
            
    with col2: 
        if st.button("Have a test of your network traffic security!"):
            switch_page('Prediction_&_Consultancy')
        
            




# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])

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
        
    

