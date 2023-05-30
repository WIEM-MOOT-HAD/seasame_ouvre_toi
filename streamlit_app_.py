import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.graph_objs as go
import missingno as msno
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




plt.rcParams.update({'font.size': 10})

data1 = pd.read_csv("compas-scores-raw.csv")
categorical_columns = ['MaritalStatus','Agency_Text', 
                       'Sex_Code_Text', 'Ethnic_Code_Text', 
                       'ScaleSet', 'AssessmentReason', 'Language', 
                       'LegalStatus', 'CustodyStatus', 'RecSupervisionLevel',
                         'RecSupervisionLevelText', 'DisplayText', 'DecileScore',
                           'ScoreText' ]


# Main Streamlit app code
def main():
    st.title('**COMPASS ANALYSIS**')

    option = st.selectbox('Please select option', ['Home Page','Data Visualisation', 'Prediction'])
    
    
    if option == "Data Visualisation":
        st.subheader('Analyse categorical columns')

        st.subheader('Descriptive statistics')
        st.write(data1.describe())

        #fig 1
        for col in categorical_columns[:-1]:
            fig, ax = plt.subplots()
            ax.hist(data1[col], edgecolor='black', color= 'lightsteelblue')
            ax.set_xticklabels(data1[col].value_counts().index, rotation=45, ha="right")
            ax.set_title(col)
            figure(figsize=(50,3))
            st.pyplot(fig)

        st.subheader('Analyse some features')


        #FIG3
        g = data1.groupby('Sex_Code_Text')
        c_female = g.get_group('Female')['DisplayText'].value_counts()
        c_male = g.get_group('Male')['DisplayText'].value_counts()

        hairpat = data1['DisplayText'].value_counts().index
        gender = data1['Sex_Code_Text'].value_counts().index
        pos = np.arange(len(hairpat))
        width = 0.35  

        fig,ax = plt.subplots()
        ax.bar(pos - width/2, c_male, width, color='lightsteelblue')
        ax.bar(pos + width/2,c_female, width, color='IndianRed')

        ax.set_xticks(pos)
        ax.set_xticklabels(hairpat, rotation=45, ha="right")

        ax.set_xlabel('DisplayText', fontsize=14)
        ax.set_title('Display Text column depending on gender',fontsize=15)
        ax.legend(gender,loc=1)
        st.pyplot(fig)


        #fig 3
        ethnic = data1.groupby('Ethnic_Code_Text')
        c_african_american= ethnic.get_group('African-American')['DisplayText'].value_counts()
        c_caucasian= ethnic.get_group('Caucasian')['DisplayText'].value_counts()
        c_hispanic= ethnic.get_group('Hispanic')['DisplayText'].value_counts()
        c_other= ethnic.get_group('Other')['DisplayText'].value_counts()
        c_asian= ethnic.get_group('Asian')['DisplayText'].value_counts()
        c_native_american = ethnic.get_group('Native American')['DisplayText'].value_counts()
        c_arabic = ethnic.get_group('Arabic')['DisplayText'].value_counts()
        c_African_Am = ethnic.get_group('African-Am')['DisplayText'].value_counts()
        c_oriental = ethnic.get_group('Oriental')['DisplayText'].value_counts()


        hairpat = data1['DisplayText'].value_counts().index
        ethnic_code = data1['Ethnic_Code_Text'].value_counts().index
        pos = np.arange(len(hairpat))
        width = 0.35  #
        fig,ax = plt.subplots()

        ax.bar(pos - width/10, c_african_american, width)
        ax.bar(pos + width/10, c_caucasian, width)
        ax.bar(pos + width/10,c_hispanic, width,)
        ax.bar(pos + width/10,c_other, width)
        ax.bar(pos + width/10,c_asian, width)
        ax.bar(pos + width/10,c_native_american, width)
        ax.bar(pos + width/10,c_arabic, width)
        ax.bar(pos + width/10,c_African_Am, width)
        ax.bar(pos + width/10,c_oriental, width)

        #plt.xticks(pos, hairpat)
        ax.set_xticks(pos, hairpat, rotation=45, ha="right")

        ax.set_xlabel('DisplayText', fontsize=14)
        ax.set_title('Histogram of risque of residivisme and ethnic origins ',fontsize=15)
        ax.legend(ethnic_code,loc=0)
        st.pyplot(fig)

        st.subheader('Correlation matrix')

        st.image('correlation_matrix.png',  use_column_width=True)

    elif option == "Prediction":
        le = preprocessing.LabelEncoder()
        cols = ["Agency_Text","Sex_Code_Text","Ethnic_Code_Text","RecSupervisionLevelText",
                "ScaleSet","Language","LegalStatus","CustodyStatus",
                  "MaritalStatus","DisplayText", "ScoreText", "AssessmentType"]
        
        st.markdown("<h1 style='font-size:24px;'>Random Forest Classifier is used to make prediction in this case </h1", unsafe_allow_html=True)
        
        st.markdown("<h1 style='font-size:24px;'> Here details of label encoded classes </h1", unsafe_allow_html=True)

        for col in cols: 
            data1[col] = le.fit_transform(data1[col])
            class_names = le.classes_
            st.write("for ", col, "encoded classes are:", class_names)
    

        clf = RandomForestClassifier()

        x = data1.drop(['Person_ID','AssessmentID', 'Case_ID','LastName',
                        'FirstName', 'MiddleName', 'ScaleSet_ID','Scale_ID', 
                        "AssessmentReason","Screening_Date","IsCompleted",
                        "IsDeleted","ScoreText", "DateOfBirth", "DecileScore", 
                        "RawScore"], axis =1) 


        y  = data1["ScoreText"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        y_predict = clf.fit(x_train, y_train).predict(x_test)

        report_df = pd.DataFrame.from_dict(classification_report(y_test, y_predict, output_dict=True)).transpose()

        cm = confusion_matrix(y_test, y_predict)
        cm_df_styled = pd.DataFrame(cm).style.background_gradient(cmap='Blues')

        st.markdown("<h1 style='font-size:24px;'>The Classification Report with a train_test_split = 0.33 </h1", unsafe_allow_html=True)

        st.write(report_df)

        st.markdown("<h1 style='font-size:24px;'> The Confusion Matrix </h1", unsafe_allow_html=True)
        st.write(cm_df_styled)

    else: 
        
        st.markdown("<h1 style='font-size:24px;'>This app show some analysis of COMPASS data set as part of Sésame, ouvre-toi project. The purpse is to detect and explain biais in model prediction </h1", unsafe_allow_html=True)
        st.image('welcome-home-minions.gif',  use_column_width=True)

# Run the app
if __name__ == '__main__':
    main()
