import streamlit as st 


st.set_page_config(
    page_title="Cyber FAQ",
    layout="wide"
)


# st.markdown(
#     "<style>#MainMenu{visibility:hidden;}</style>",
#     unsafe_allow_html=True
# )

# au.render_cta()

st.title("FAQ")


st.markdown("#### Here are some information that might help you:")



    
with st.expander("What is Cyberattack", expanded=False):
    
        st.markdown(f"[{'Cyberattack'}]({'https://en.wikipedia.org/wiki/Cyberattack'})", unsafe_allow_html=True)
                
                
with st.expander("Types of Cyberattacks"):
    st.markdown(f"[{'Denial-of-service attack'}]({'https://en.wikipedia.org/wiki/Denial-of-service_attack'})", unsafe_allow_html=True)
    st.markdown(f"[{'Man-in-the-middle attack'}]({'https://en.wikipedia.org/wiki/Man-in-the-middle_attack'})", unsafe_allow_html=True)
    st.markdown(f"[{'Code injection'}]({'https://en.wikipedia.org/wiki/Code_injection'})", unsafe_allow_html=True)
    st.markdown(f"[{'Malware'}]({'https://en.wikipedia.org/wiki/Malware'})", unsafe_allow_html=True)
    st.markdown(f"[{'Ransomware'}]({'https://en.wikipedia.org/wiki/Ransomware'})", unsafe_allow_html=True)
    st.markdown(f"[{'TCP/IP stack fingerprinting'}]({'https://en.wikipedia.org/wiki/TCP/IP_stack_fingerprinting'})", unsafe_allow_html=True)
    st.markdown(f"[{'Port scanner'}]({'https://en.wikipedia.org/wiki/Port_scanner'})", unsafe_allow_html=True)




    
st.markdown("#### OpenAI related")

with st.expander("What is an OpenAI API Key and why do I need one?"):
    st.markdown("An OpenAI API key is a unique credential that allows you to interact with OpeAI's GPT models. It also serves as your identifier in GPT Lab, allowing us to remember the AI Assistants you have created.")

with st.expander("How can I get an OpenAI API Key?"):
    st.markdown("You can obtain an OpenAI API Key by creating one on the OpenAI website: https://platform.openai.com/account/api-keys")
    


with st.expander("Do you have some recommendations on how to create good prompts?"):
    st.markdown("""
    Yes, here are few tips to creating effective prompts:  \n
    * Familiarize yourself with the best practices for prompt engineering, as outlined in this OpenAI article: https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api  \n
    * When creating a prompt for a GPT Lab AI Assistant, make sure to include instructions for the Assistant to introduce itself to the user first. This helps ensure a smooth and engaging chat session.  
    * Test out your prompt in the Lab to ensure it accurately conveys the desired topic or task.   
    """)
