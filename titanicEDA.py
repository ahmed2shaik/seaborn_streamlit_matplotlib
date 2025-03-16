import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
@st.cache_data
def load_data():
    data = pd.read_csv(r'D:\FSDS\Datasets\titanic\train.csv')
    return data

data=load_data()

# Title and Description
st.title('Exploratory Data Analysis of Titanic Dataset')
st.write('This is an EDA on Titanic Dataset')
st.write('First few rows of the dataset:')
st.dataframe(data.head())

# Data cleaning section
st.subheader('Missing Values')
missing_data=data.isnull().sum()
st.write(missing_data)

if st.checkbox('Fill missing Age with mean'):
    # data['Age'] = data['Age'].fillna(data['Age'].median, inplace=True)
    data['Age'] = data['Age'].fillna(data['Age'].mean())
if st.checkbox('Fill missing Embarked with mode'):
    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)

if st.checkbox('Drop Duplicates'):
    data.drop_duplicates(inplace=True)

st.subheader('Cleaned Dataset')    
st.dataframe(data.head())

# EDA Section
st.subheader('Statistical Summary of the data')
st.write(data.describe())

# Age Distribution
st.subheader('Age Distribution')
fig,ax = plt.subplots()
sns.histplot(data['Age'], kde=True, ax=ax)
ax.set_title('Age Distribution')
st.pyplot(fig)

# Gender Distribution
st.subheader('Gender Distribution')
fig,ax = plt.subplots()
sns.countplot(x='Sex', data=data , ax=ax)
ax.set_title('Gender Distribution')
st.pyplot(fig)

# PClass vs Survived
st.subheader('PClass vs Survived')
fig,ax = plt.subplots()
sns.countplot(x='Pclass', data=data ,ax=ax)
ax.set_title('PClass vs Survived')
st.pyplot(fig)

# correlation heatmap
st.subheader('correlation heatmap')
# Select numerical columns for correlation matrix
numerical_columns = data.select_dtypes(include='number')

# Calculate the correlation matrix
corr_matrix = numerical_columns.corr()
fig,ax = plt.subplots()
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',ax=ax)
ax.set_title('correlation heatmap')
st.pyplot(fig)

# Feature Engineering Section
st.subheader('Feature Engineering Section')
data['FamilySize'] = data['SibSp'] +  data['Parch']
fig,ax = plt.subplots()
sns.histplot(data['FamilySize'],kde=True,ax=ax)
ax.set_title('Family Size Distribution')
st.pyplot(fig)

# Conclusion Section
st.subheader('Key Insights')
insights="""
- Females have a higher surival rate than males.
- Passangers in 1st class have higher surival rate.
- The majority of passangers are in Pclass 3.
- Younger Passanger tends to survive most 
"""
st.write(insights)