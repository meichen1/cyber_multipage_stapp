import pickle
import streamlit as st
import io
import pandas as pd
from helper import cutCatTranformer, columnDropperTransformer, AttackTypeMapping, Prediction_Report
from sklearn.model_selection import train_test_split
# loading the trained model
trained_model = 'trained_model/model_xgboost15class.pkl'
model = pickle.load(open(trained_model, 'rb'))

#@st.cache_data(experimental_allow_widgets=True) 

def main():
    
    ## write a front end view for the app
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Cyber Attack Detector</h2>
    </div>
    """
    ## display the front end aspect
    st.markdown(html_temp,unsafe_allow_html=True)
    
    st.write('❤️ This is a simple app to show the results of the predictive maintenance model.')
    
    # upload a file
    file = st.file_uploader("Upload file", type=["csv"])
    if file is not None:
        try:
            ## read the uploaded csv file          
            df = pd.read_csv(file, low_memory=False)
            X,y = df.drop(columns=['Attack_label', 'Attack_type']), df['Attack_type']
            st.subheader('Data')
            st.write(df.head())
           
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
            values = st.slider(
                'Select the percentage of data you want for model prediction',
                0.0, 100.0, (20.0))
            st.write(values,'%')

        except Exception as e:
            st.write(str(e))
            
    if st.button('Predict'):
        try:
            X_train, X, y_train, y = train_test_split(X, y, test_size=values/100.0)

            # set the selected rows for prediction
            # selected_indices = edited_df.loc[edited_df["select"] == True].index.values
            # X = X.loc[selected_indices]
            # y = y.loc[selected_indices]


            # call labelEncoder_y.encode in the helper.py to encode the 15 multiclasses
            attackTypeMapping = AttackTypeMapping()
            y = attackTypeMapping.map_type2value(y)

            predictions = model.predict(X)

            # print the prediction (first 5 lines)
            prediction_output = pd.DataFrame(attackTypeMapping.map_value2type(predictions), columns = ['attack_type_prediction'])
            prediction_output = pd.concat([prediction_output, X.reset_index(drop=True)], axis=1)
            st.write(prediction_output.head())

            # report 
            prediction_report = Prediction_Report()
            prediction_report.report_precision_recall(y, predictions)
            fig = prediction_report.plot_confusion_matrix(y, predictions, attackTypeMapping)
            st.pyplot(fig)
            
        except Exception as e:
            st.write(str(e))
    


if __name__ == '__main__':
    main()
    
    