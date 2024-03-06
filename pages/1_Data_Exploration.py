
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from src.helper import cutCatTranformer, columnDropperTransformer, AttackTypeMapping, Prediction_Report
from Home import load_data



st.set_page_config(
    page_title="Cyber Data Exploration",
    layout="wide"
)

# alt.themes.enable("dark")

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
        
        attackTypeMapping = AttackTypeMapping()
        y = attackTypeMapping.map_type2value(y)

        
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
        

        st.subheader('EDA')
                
        if st.button('Analyze data'):
            
            # st.subheader('🤔 Correlations of numerical features')
            # # selected_numerical_features = ['icmp.seq_le', 'icmp.checksum', 'http.content_length', 'tcp.connection.rst', 'tcp.ack', 'mqtt.topic_len']
            # # correlation_matrix = X[selected_numerical_features].corr()
            # numerical_features = ['arp.opcode', 'icmp.checksum', 'icmp.seq_le', 'icmp.transmit_timestamp',
            #    'http.content_length', 'http.response', 'tcp.ack', 'tcp.ack_raw',
            #    'tcp.checksum', 'tcp.connection.fin', 'tcp.connection.rst',
            #    'tcp.connection.syn', 'tcp.connection.synack', 'tcp.dstport',
            #    'tcp.flags', 'tcp.len', 'tcp.seq', 'udp.port', 'udp.stream',
            #    'udp.time_delta', 'dns.qry.name', 'dns.qry.qu', 'dns.retransmission',
            #    'dns.retransmit_request', 'mqtt.conflag.cleansess', 'mqtt.hdrflags',
            #    'mqtt.len']
            # st.write('There are', len(numerical_features), 'numerical features')
            # correlation_matrix = X[numerical_features].corr()
            
            # fig = px.imshow(correlation_matrix, title='Correlation Matrix of Numerical Features')
            # st.plotly_chart(fig)
            
            # col1, col2 = st.columns(2)
            # df_ml_num_type = pd.concat([X[numerical_features], y], axis=1)
            # le = LabelEncoder()
            # df_ml_num_type['Attack_type'] = le.fit_transform(df_ml_num_type['Attack_type'])
            
            # i=0
            # for column in df_ml_num_type.columns[:-1]:
            #     fig = go.Figure()
            #     fig.add_trace(go.Histogram(x=df_ml_num_type[column], name=column))
            #     fig.update_layout(
            #         title_text=f'{column} Distribution',
            #         xaxis_title_text='Value',
            #         yaxis_title_text='Count',
            #         bargap=0.2,
            #         bargroupgap=0.1
            #     )
            #       ## sequentially add the pie plot to the 3 columns
            #     i = i % 2
            #     if i == 0:
            #         col1.plotly_chart(fig)
            #     elif i == 1:
            #         col2.plotly_chart(fig)
            #     i += 1
            #     st.plotly_chart(fig)
            
            
            
            
            
            st.subheader('🤓 Selected categorical features')
            ## have 3 columns to display the histogram of categorical features
            col1, col2 = st.columns(2)
            
            # selected_categorical_features = ['mqtt.protoname', 'http.response', 'http.request.method', 'mqtt.conack.flags']
            categorical_features = ['http.request.method', 'http.referer', 'http.response', 'mqtt.conack.flags', 'mqtt.protoname','mqtt.topic', 'http.request.version', 'dns.qry.name.len']
            
            ## pie plot top 10 frequent categories for each categorical feature, with all the other categories combined into 'Others'
            i = 0
            for col in categorical_features:
                
                fig = px.pie(X[col].value_counts().iloc[:10], names=X[col].value_counts().iloc[:10].index, values=X[col].value_counts().iloc[:10].values, title=col,)
                fig.update_layout(showlegend = False)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                ## set the layout compact and set figure size 1/3 of the screen
                fig.update_layout(
                    autosize=False,
                    width=300,
                    height=300
                )
                ## sequentially add the pie plot to the 3 columns
                i = i % 2
                if i == 0:
                    col1.plotly_chart(fig)
                elif i == 1:
                    col2.plotly_chart(fig)
                i += 1
            # st.subheader('Features vs Attack Type')
            
            
                