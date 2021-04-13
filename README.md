# Airplane_fair_prediction
#Predicting Fare of airline Tickets using ML

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#read data
train_data=pd.read_excel('Data_Train.xlsx')
train_data.head()
train_data.shape
#data cleaning and data preprocessing
train_data.isna().sum()#to se how many missing values are there
train_data.dropna(inplace=True)#to drop those missing values
train_data.isna().sum()
train_data.dtypes#check datatype of all columns

#changing date and time datatype
def change_into_datetime(col):
    train_data[col]=pd.to_datetime(train_data[col])
    
train_data.columns
for i in ['Date_of_Journey','Dep_Time','Arrival_Time']:
    change_into_datetime(i)    
train_data.dtypes

train_data['journey_day']=train_data['Date_of_Journey'].dt.day#making separate column for month and date
train_data['journey_month']=train_data['Date_of_Journey'].dt.month
train_data.head()
train_data.drop('Date_of_Journey',axis=1,inplace=True)
train_data.head()

#separate hour & minute in departure & arrival
def extract_hour(df,col):
    df[col+'_hour']=df[col].dt.hour

def extract_min(df,col):
    df[col+'_minute']=df[col].dt.minute
    
extract_hour(train_data,'Dep_Time')
extract_min(train_data,'Dep_Time')
train_data.drop('Dep_Time',axis=1,inplace=True)
train_data.head()

extract_hour(train_data,'Arrival_Time')
extract_min(train_data,'Arrival_Time')
train_data.drop('Arrival_Time',axis=1,inplace=True)
train_data.head()

#splitting out duration into hours & min
duration=list(train_data['Duration'])
for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]:
            duration[i]=duration[i] + ' 0m'
        else:
            duration[i]='0h ' + duration[i]
train_data['Duration']= duration

train_data.head()

def hour(x):
    return x.split(' ')[0][0:-1]

def minute(x):
    return x.split(' ')[1][0:-1]

train_data['Duration_hours']=train_data['Duration'].apply(hour)
train_data['Duration_mins']=train_data['Duration'].apply(minute)
train_data.head()
train_data.drop('Duration',axis=1,inplace=True)
train_data.columns
train_data.dtypes

#changing the data type of duration_hours % duration_min
train_data['Duration_hours']=train_data['Duration_hours'].astype(int)
train_data['Duration_mins']=train_data['Duration_mins'].astype(int)
train_data.dtypes

#Checking for categorical & continous data
cat_col=[col for col in train_data.columns if train_data[col].dtypes=='O']
cat_col  #O is for object giving categorical data

cont_col=[col for col in train_data.columns if train_data[col].dtypes!='O']
cont_col  #!= O is for not categorical data which is continous
 
#nominal & ordinal data
categorical=train_data[cat_col]
categorical.head()  #using onehot encoding
categorical['Airline'].value_counts() #gives details of airline column
plt.figure(figsize=(15,5))  #boxplot size
sns.boxplot(x='Airline', y='Price', data=train_data.sort_values('Price',ascending=False))
sns.boxplot(x='Total_Stops', y='Price', data=train_data.sort_values('Price',ascending=False))
#changing categoricL data of Airline column into 0 & 1
Airline=pd.get_dummies(categorical['Airline'],drop_first=True)
Airline.head()  #changed rows into 0 & 1

#plot of source column using onehot coding
categorical['Source'].value_counts() #gives details of airline column
plt.figure(figsize=(15,5))  #boxplot size
sns.boxplot(x='Source', y='Price', data=train_data.sort_values('Price',ascending=False))
Source=pd.get_dummies(categorical['Source'],drop_first=True)
Source.head()

#plot of destination using onehot encoding
categorical['Destination'].value_counts() #gives details of airline column
plt.figure(figsize=(15,5))  #boxplot size
sns.boxplot(x='Destination', y='Price', data=train_data.sort_values('Price',ascending=False))
Destination=pd.get_dummies(categorical['Destination'],drop_first=True)
Destination.head()

#LabelEncoder encoding on Route column
categorical['Route']#warning will be popped ignore it move ahead
categorical['Route_1']=categorical['Route'].str.split('→').str[0]
categorical['Route_2']=categorical['Route'].str.split('→').str[1]
categorical['Route_3']=categorical['Route'].str.split('→').str[2]
categorical['Route_4']=categorical['Route'].str.split('→').str[3]
categorical['Route_5']=categorical['Route'].str.split('→').str[4]

categorical.head()
#dropping route
categorical.drop('Route',axis=1,inplace=True)  #use drop with categorical and not train_data
categorical.isnull().sum()  #checking for null values ,v c route 3,4,5 havae many 
categorical.head()

#replace null values with none
for i in ['Route_3','Route_4','Route_5']:
    categorical[i].fillna('None',inplace=True)
categorical.isnull().sum()  #checking for null values ,v c route 3,4,5 havae many 

for i in categorical.columns:
    print('{} has total {} categories'.format(i,len(categorical[i].value_counts() )))
    
#importing labelencoder library
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
categorical.columns
for i in ['Route_1', 'Route_2', 'Route_3', 'Route_4','Route_5']:
    categorical[i]=encoder.fit_transform(categorical[i])
categorical.head()

#drop additional_info column as no much info is given
categorical.drop('Additional_Info',axis=1,inplace=True)
categorical.head()

#data pre-processing on Total-stops 
categorical['Total_Stops'].unique()  #giving u values of particular column
#defining dict to replace names eg.2 stops by 2
dict={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}
#connecting that with categorical using map
categorical['Total_Stops']=categorical['Total_Stops'].map(dict)
categorical.head()
#joining all columns
data_train=pd.concat([categorical,Airline,Source,Destination,train_data[cont_col]],axis=1)
data_train.head()

#show all column
pd.set_option('display.max_columns',38)
data_train.head()

#visualizing outliers by two plot:distribution approach & boxplot
def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)  #2rows,1column
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)
plot(data_train,'Price')
#with condition finding out median 
data_train['Price']=np.where(data_train['Price']>=40000,data_train['Price'].median(),data_train['Price'])
plot(data_train,'Price')

#separate independent & dependent feature
#x variable will contain all independent feature
#y variable will contain all dependent feature
X=data_train.drop('Price',axis=1)
X.head()
X.shape
y=data_train['Price']
y
data_train['IndiGo']#to check details of particular column
#apply feature selection
from sklearn.feature_selection import mutual_info_classif
mutual_info_classif(X,y)#when running this line error comes which says Valuerror: str cannot be converted o float:IndiGo
#create datframe of all coulmn with respect to its priority
#imp=pd.DataFrame(mutual_info_classif(X,y),index=X.columns)
#imp
#rename coulmn names with priority & in ascending order
#imp.columns=['importance']
#imp.sort_values(by='importance',ascending=False)















