from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


class cutCatTranformer():
    '''
    input pd.dataframe, cut categories with count less than threshold into one category 'Others'.
    '''
    
    def __init__(self, threshold=3):
        self.threshold = threshold
        
    def fit(self, X):
        self.columns = X.columns
        self.lst = []
        for col in X.columns:
            self.lst.append(Counter(X[col].values))
               
    def transform(self, X, y=None):
        Xcopy = X.copy()
        for i, col in enumerate(X.columns):
            Xcopy[col] = X[col].mask(X[col].map(self.lst[i]) < self.threshold, 'Others')
        return Xcopy
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class columnDropperTransformer():
    '''
    Drop columns with values all 0
    '''
    def __init__(self):
      self.columns = []
      self.features = []
      pass

    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        # return columns name that all values are 0
        zero_columns = [col for col in X.columns if X[col].value_counts().get(0) == len(X)]
        self.columns = zero_columns
        # remaining columns name
        non_zero_columns = [col for col in X.columns if col not in zero_columns]
        self.features = non_zero_columns
        return

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.features

# class LabelEncoder_y():
#     '''
#     label encoder for y (15 classes)
#     '''
#     def __init__(self):
#         self.label_encoder = LabelEncoder()

#     def encode(self,y):
#         # encode Y class values as integers
#         self.label_encoder.fit(y)
#         return
    
#     def transform(self,y):
#         return self.label_encoder.transform(y)
    
#     def decode(self, prediction):
#         class_mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
#         class_names = list(class_mapping.keys())
#         prediction_names = [class_names[i] for i in prediction]
#         # print(prediction_names)
    
#     def get_encoder(self):
#         return self.label_encoder
    
class AttackTypeMapping():
    '''
    map the 15 classes to values
    '''
    def __init__(self):
      self.Attack_type_mapping = {'Backdoor': 0,
                              'DDoS_HTTP': 1,
                              'DDoS_ICMP': 2,
                              'DDoS_TCP': 3,
                              'DDoS_UDP': 4,
                              'Fingerprinting': 5,
                              'MITM': 6,
                              'Normal': 7,
                              'Password': 8,
                              'Port_Scanning': 9,
                              'Ransomware': 10,
                              'SQL_injection': 11,
                              'Uploading': 12,
                              'Vulnerability_scanner': 13,
                              'XSS': 14}

    def map_type2value(self,y):
        return y.map(self.Attack_type_mapping)

    def map_value2type(self,prediction):
        class_names = list(self.Attack_type_mapping.keys())
        prediction_names = [class_names[i] for i in prediction]
        return prediction_names
    
    def get_mapping(self):
      return self.Attack_type_mapping


class Prediction_Report():
    def __init__(self):
        pass

    '''
    def get_max_prediction_proba(self, predictions):
        vals = []
        for pred in predictions:
            max_index = np.where(pred == np.amax(pred))[0][0]
            vals.append(max_index)
        return vals
    '''

    def report_precision_recall(self,y_test, y_pred):
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
        return format(precision, ".2f"), format(recall, ".2f"), format(fscore, ".2f")

        
    def plot_confusion_matrix(self, y_test, y_pred, attackTypeMapping):
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_names = list(attackTypeMapping.get_mapping().keys())
        # print(conf_matrix)

        sns.set_theme(font_scale=0.5, font="sans-serif")
        plt.figure(figsize=(4, 4))
        fig = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=False, 
                          xticklabels=class_names, yticklabels=class_names).get_figure() #,xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_
        plt.xlabel("Predicted Cyber Attack Type")
        plt.ylabel("True Cyber Attack Type")
        plt.title("Confusion Matrix")
        plt.show()

        return fig

    