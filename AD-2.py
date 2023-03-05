#!/usr/bin/env python
# coding: utf-8

# # Crop Production In india 

# In[1]:


##Importing important libraries.
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")


# In[2]:


##Import Data
df=pd.read_csv("crop_production.csv")
df.head()


# In[3]:


df.shape      


# #There are 246091 rows and 7 Columns of attribute.

# In[4]:


df.info()       


# #There is 7 attributes, in which four (state_Name,District_Name,Season,Crop) are categorical and three are numerical attributes.

# In[5]:


df.dtypes


# In[6]:


df.State_Name.nunique() 


#  ##Total number of state is 33 including union territory

# In[7]:


df.State_Name.unique()


# In[8]:


df.Season.unique() ##Total no of season 


# ##There are trailing white spaces in some entries

# In[9]:


df.Crop.nunique()


# In[10]:


# To remove white space at the end of strings
df["Season"] = df["Season"].str.rstrip()
df["State_Name"] = df["State_Name"].str.rstrip()
df["Crop"] = df["Crop"].str.rstrip() 
df["District_Name"] = df["District_Name"].str.rstrip() 
df.Season.unique()


# In[11]:


print(df.Crop_Year.nunique())
print(df.Crop_Year.unique())


# #The Data has been collected from the year 1997 to 2015 (19 yrs)

# In[12]:


df.District_Name.nunique() ## No. of districts 


# In[13]:


df.isnull().sum()    ##Checking for missing values ,there is 3730 vaues in production are missing.


# In[14]:


df['Production'] = df['Production'].replace(0, np.nan)
df['Production']=df['Production'].fillna(df.groupby('Crop')['Production'].transform('mean'))


# In[15]:


df.isnull().sum() 


# In[16]:


Crop_Prod = df.groupby('Crop')['Production'].sum().reset_index().sort_values(by='Production',ascending=False)
Crop_Prod.tail(20) 


# In[17]:


##the percentage of missing values is very less compare to whole data so we can drop the missing values rows
df.dropna(subset=["Production"],axis=0,inplace=True) 


# In[18]:


df.isnull().sum() 
## now there is  no missing values after dropping the missing values rows.


# In[19]:


df.info() 


#  #rows reduced to 238838 after dropping.

# In[20]:


df.District_Name.value_counts() 
# shows the no. of data from each districts


# # Add Rainfall Data

# In[21]:


# import rainfall data
raindata=pd.read_csv("rainfall in india.csv")
raindata.head()


# In[22]:


raindata["YEAR"].unique()


# In[23]:


raindata.isnull().sum()


# In[24]:


raindata.rename(columns={"STATE":"State_Name","YEAR":"Crop_Year","ANNUAL RAINFALL":"Rainfall"},inplace = True)


# In[25]:


print(raindata.State_Name.unique())
print(df.State_Name.unique())


# In[26]:


raindata["State_Name"].replace({'ARUNACHAL PRADESH':'Arunachal Pradesh',
                    'ORISSA':'Odisha', 'JHARKHAND': 'Jharkhand','BIHAR':'Bihar',
                    'UTTARAKHAND':'Uttarakhand',  'PUNJAB':'Punjab','HIMACHAL PRADESH':'Himachal Pradesh',
                    'JAMMU & KASHMIR':'Jammu and Kashmir', 'CHHATTISGARH':'Chhattisgarh', 'TELANGANA':'Telangana',
                    'TAMIL NADU':"Tamil Nadu", 'KERALA':'Kerala','ANDAMAN & NICOBAR ISLANDS': 'Andaman and Nicobar Islands',
                    'ASSAM':'Assam','MEGHALAYA':'Meghalaya', 'HARYANA ':'Haryana', 'CHANDIGARH':'Chandigarh',
                    'NAGALAND':'Nagaland', 'MANIPUR':'Manipur', 'MIZORAM ':'Mizoram', 'TRIPURA':'Tripura', 
                    'PUDUCHERRY':'Puducherry', 'UTTAR PRADESH': 'Uttar Pradesh', 'RAJASTHAN':'Rajasthan',
                    'MADHYAPRADESH':'Madhya Pradesh','SIKKIM':'Sikkim', 'GUJARAT':'Gujarat', 'GOA': 'Goa',
                    'ANDHRA PRADESH':'Andhra Pradesh', 'KARNATAKA':'Karnataka', 'WEST BENGAL':'West Bengal',
                    'MAHARASHTRA':'Maharashtra','DADRA NAGAR HAVELI':'Dadra and Nagar Haveli'}, inplace=True)
raindata.State_Name.unique()


# In[27]:


set(raindata.State_Name)==set(df.State_Name)


# In[28]:


raindata=raindata[raindata.State_Name!="LAKSHADWEEP"]


# In[29]:


set(raindata.State_Name)==set(df.State_Name)


# In[30]:


raindata.tail(10)


# In[31]:


df = pd.merge(df, raindata)


# In[32]:


df.head()


# In[33]:


df=df[["State_Name","District_Name","Crop_Year","Season","Area","Rainfall","Crop","Production"]]
df.head()


# In[34]:


df.describe()


# In[35]:


sns.heatmap(df.corr(),annot=True)


# # State

# In[36]:


df.State_Name.value_counts().head(15)  


# #This shows that we have more data from rich agriculture state like UP,MP etc..

# In[37]:


state_prod = df.groupby('State_Name')['Production'].sum().reset_index().sort_values(by='Production',ascending=False)


# In[38]:


plt.figure(figsize=(10,10))
plt.xticks(rotation=90)
sns.barplot(data=state_prod, y= "State_Name",x="Production")


# In[39]:


state_prod.head(5) # Top 5 states with most production over the years


# In[40]:


state_prod.tail(5) # Top 5 states with least production over the years


# #We can see that Kerela has the highest production, 
# Also we can see that the top 3 states with highest production over the years are from South India

# # Crop Year

# In[41]:


print(df.Crop_Year.value_counts())  ## count of data in each year


# In[42]:


plt.figure(figsize=(12,8),dpi=100)
sns.barplot(data=df,x='Crop_Year',y='Production');


# In[43]:


plt.figure(figsize=(9,7),dpi=100)
sns.lineplot(data=df,x='Crop_Year',y='Production');


# #the period between 2010-2012 and then 2012-2013 happens to be the year which saw highest yield for the crops.

# In[44]:


#  production from each state in a year 
prod_year= df.groupby(['State_Name','Crop_Year'])['Production'].sum().reset_index().sort_values(by='Production',ascending=False)
prod_year.head(10)  # Top states with most production in a year


# In[45]:


prod_year.tail(10)


# # Season

# In[46]:


plt.figure(figsize=(7,5),dpi=100)
sns.countplot(data=df,x='Season');


# In[47]:


plt.figure(figsize=(7,5),dpi=100)
sns.barplot(data=df,x='Season',y='Production');


# #Whole Year season seems to have yeilded more crops compared to other seasons

# ## Crops

# In[48]:


print(df.Crop.value_counts().head(10))


# In[49]:


Crop_Prod = df.groupby('Crop')['Production'].sum().reset_index().sort_values(by='Production',ascending=False)
Crop_Prod.head(10) 


# In[50]:


plt.figure(figsize=(12,5),dpi=100)
sns.barplot(data=Crop_Prod.head(15),x='Production',y='Crop')


# #From 1997 to 2015,Rice is produced more, but more yield is obtained from coconut

# #Coconut, Sugarcane, Rice, Wheat and Potato happen to be the top 5 crops yeilding more productions in India over the years

# In[51]:


Crop_Prod.tail(10)  # worst performing crops


# # Analyze the highest produced crops in the country

# # 1. Coconut

# In[52]:


coconut_df = df[df['Crop'] == 'Coconut']


# In[53]:


# Average Coconut production in states
plt.figure(figsize=(8,5),dpi=100)
sns.barplot(data=coconut_df,x='Production',y='State_Name');


# In[54]:


coconut_df.groupby('State_Name').sum()['Production'].nlargest().reset_index()


# #Kerala followed by Andra Pradesh and Andaman & Nicobar islands has highest production of coconut in India

# In[55]:


# Coconut Production vs Year
plt.figure(figsize=(8,5),dpi=100)
sns.lineplot(data=coconut_df,x='Crop_Year',y='Production');


# #We can see that, coconut production is increasing over years

# In[56]:


# cocnut production vs Season
coconut_df.groupby('Season')['Production'].sum().nlargest()


# In[57]:


sns.heatmap(coconut_df.corr(),annot=True,cmap = "coolwarm" )


# #Whole Year season has the highest yield of Cococnut crops

# In[58]:


sns.scatterplot(data=coconut_df,x='Area',y='Production');


# In[59]:


# coconut production in districts 
plt.figure(figsize=(10,5),dpi=100)
sns.barplot(data=coconut_df.groupby('District_Name').sum()['Production'].nlargest().reset_index(),x='District_Name',y='Production')


# # 2. Sugar cane

# In[60]:


sugar_df = df[df['Crop'] == 'Sugarcane']


# In[61]:


# Average Sugar cane production in states
plt.figure(figsize=(8,7),dpi=100)
sns.barplot(data=sugar_df,x='Production',y='State_Name')


# In[62]:


sugar_df.groupby('State_Name').sum()['Production'].nlargest().reset_index()


# #UP, Maharashtra, Tamil Nadu, Karnataka and Andra Pradesh have the highest production of Sugarcane

# In[63]:


# Sugarcane production in districts 
plt.figure(figsize=(10,5),dpi=100)
sns.barplot(data=sugar_df.groupby('District_Name').sum()['Production'].nlargest().reset_index(),x='District_Name',y='Production')


# In[64]:


plt.figure(figsize=(8,6),dpi=100)
sns.lineplot(data=sugar_df,x='Crop_Year',y='Production');


# #Production of sugarcane decreased drastically during 1998, and then it was stable till 2014

# In[66]:


sns.barplot(data=sugar_df,x='Season',y='Production')


# In[67]:


sns.heatmap(sugar_df.corr(),annot=True,cmap = "coolwarm" )


# In[68]:


sns.scatterplot(data=sugar_df,x='Area',y='Production');


# # 3. Rice

# In[69]:


rice_df = df[df['Crop'] == 'Rice']


# In[70]:


rice_df.groupby('State_Name').sum()['Production'].nlargest().reset_index()


# In[71]:


# Average Rice production in states
plt.figure(figsize=(8,7),dpi=100)
sns.barplot(data=rice_df,x='Production',y='State_Name')


# In[72]:


plt.figure(figsize=(12,6),dpi=90)
sns.barplot(data=rice_df,x='Crop_Year',y='Production');


# In[73]:


sns.barplot(data=rice_df,x='Season',y='Production')


# In[74]:


sns.heatmap(rice_df.corr(),annot=True,cmap = "coolwarm" )


# In[75]:


sns.scatterplot(data=rice_df,x='Area',y='Production');


# # Zone-Wise Production - 1997-2014

# In[76]:



north_india = ['Jammu and Kashmir', 'Punjab', 'Himachal Pradesh', 'Haryana', 'Uttarakhand', 'Uttar Pradesh', 'Chandigarh']
east_india = ['Bihar', 'Odisha', 'Jharkhand', 'West Bengal']
south_india = ['Andhra Pradesh', 'Karnataka', 'Kerala' ,'Tamil Nadu', 'Telangana']
west_india = ['Rajasthan' , 'Gujarat', 'Goa','Maharashtra']
central_india = ['Madhya Pradesh', 'Chhattisgarh']
north_east_india = ['Assam', 'Sikkim', 'Nagaland', 'Meghalaya', 'Manipur', 'Mizoram', 'Tripura', 'Arunachal Pradesh']
ut_india = ['Andaman and Nicobar Islands', 'Dadra and Nagar Haveli', 'Puducherry']


# In[77]:


def get_zonal_names(row):
    if row['State_Name'].strip() in north_india:
        val = 'North Zone'
    elif row['State_Name'].strip()  in south_india:
        val = 'South Zone'
    elif row['State_Name'].strip()  in east_india:
        val = 'East Zone'
    elif row['State_Name'].strip()  in west_india:
        val = 'West Zone'
    elif row['State_Name'].strip()  in central_india:
        val = 'Central Zone'
    elif row['State_Name'].strip()  in north_east_india:
        val = 'NE Zone'
    elif row['State_Name'].strip()  in ut_india:
        val = 'Union Terr'
    else:
        val = 'No Value'
    return val

df['Zones'] = df.apply(get_zonal_names, axis=1)
df['Zones'].unique()


# In[78]:


df.Zones.value_counts()


# In[79]:


fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(x=df.Zones.sort_values(ascending=True), y=df.Production)
plt.yscale('log')
plt.title('Zone-Wise Production: Total')


# # Different categories of Crops

# In[80]:


crop=df['Crop']
def cat_crop(crop):
    for i in ['Rice','Maize','Wheat','Barley','Varagu','Other Cereals & Millets','Ragi','Small millets','Bajra','Jowar', 'Paddy','Total foodgrain','Jobster']:
        if crop==i:
            return 'Cereal'
    for i in ['Moong','Urad','Arhar/Tur','Peas & beans','Masoor',
              'Other Kharif pulses','other misc. pulses','Ricebean (nagadal)',
              'Rajmash Kholar','Lentil','Samai','Blackgram','Korra','Cowpea(Lobia)',
              'Other  Rabi pulses','Other Kharif pulses','Peas & beans (Pulses)','Pulses total','Gram']:
        if crop==i:
            return 'Pulses'
    for i in ['Peach','Apple','Litchi','Pear','Plums','Ber','Sapota','Lemon','Pome Granet',
               'Other Citrus Fruit','Water Melon','Jack Fruit','Grapes','Pineapple','Orange',
               'Pome Fruit','Citrus Fruit','Other Fresh Fruits','Mango','Papaya','Coconut','Banana']:
        if crop==i:
            return 'Fruits'
    for i in ['Bean','Lab-Lab','Moth','Guar seed','Soyabean','Horse-gram']:
        if crop==i:
            return 'Beans'
    for i in ['Turnip','Peas','Beet Root','Carrot','Yam','Ribed Guard','Ash Gourd ','Pump Kin','Redish','Snak Guard','Bottle Gourd',
              'Bitter Gourd','Cucumber','Drum Stick','Cauliflower','Beans & Mutter(Vegetable)','Cabbage',
              'Bhindi','Tomato','Brinjal','Khesari','Sweet potato','Potato','Onion','Tapioca','Colocosia']:
              if crop==i:
                return 'Vegetables'
    for i in ['Perilla','Ginger','Cardamom','Black pepper','Dry ginger','Garlic','Coriander','Turmeric','Dry chillies','Cond-spcs other']:
        if crop==i:
            return 'spices'
    for i in ['other fibres','Kapas','Jute & mesta','Jute','Mesta','Cotton(lint)','Sannhamp']:
        if crop==i:
            return 'fibres'
    for i in ['Arcanut (Processed)','Atcanut (Raw)','Cashewnut Processed','Cashewnut Raw','Cashewnut','Arecanut','Groundnut']:
        if crop==i:
            return 'Nuts'
    for i in ['other oilseeds','Safflower','Niger seed','Castor seed','Linseed','Sunflower','Rapeseed &Mustard','Sesamum','Oilseeds total']:
        if crop==i:
            return 'oilseeds'
    for i in ['Tobacco','Coffee','Tea','Sugarcane','Rubber']:
        if crop==i:
            return 'Commercial'

df['cat_crop']=df['Crop'].apply(cat_crop)


# In[81]:


df["cat_crop"].value_counts()


# In[82]:


plt.figure(figsize=(15,8))
plt.tick_params(labelsize=10)
df.groupby("cat_crop")["Production"].agg("count").plot.bar()
plt.show()


# In[83]:


plt.figure(figsize=(12,6),dpi=90)
sns.barplot(data=df,x='cat_crop',y='Production');


# In[84]:


cultivation_data = df[["State_Name","District_Name","Crop_Year","Season","Rainfall","Crop","Area","Production"]]
cultivation_data.head()


# # Dummy encoding

# In[85]:


cultivation_data = pd.get_dummies(cultivation_data)
x = cultivation_data.drop("Production",axis=1)
y = cultivation_data[["Production"]]


# # Train/Test Split

# In[86]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=42)
print("x_train :",x_train.shape)
print("x_test :",x_test.shape)
print("y_train :",y_train.shape)
print("y_test :",y_test.shape)


# # Regression Models

# In[87]:


Predictions=y_test.copy()
models = []
models.append(('dtr_reg', DecisionTreeRegressor(random_state=42)))
models.append(('knr_reg', KNeighborsRegressor()))
models.append(('bagging_reg', BaggingRegressor(n_estimators=10, random_state=42)))
models.append(('rf_reg', RandomForestRegressor(n_estimators=10,random_state=42)))
models.append(('grad_reg', GradientBoostingRegressor()))

# evaluate each model in turn
results = []
names=[]
RMSE=[]
for name,model in models:
    bla=model.fit(x_train, y_train)
    print(f"%s === "%model)
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    kf_cv_scores = cross_val_score(model, x_train, y_train, cv=kfold)
    for i in range (len(kf_cv_scores)):
        print(f"%s {[i]} Accuracy :  {kf_cv_scores[i]}"% model)
    
    print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
    results.append(kf_cv_scores)
    names.append(name)
    ypred = model.predict(x_test)
    Predictions[name]=ypred
    mse = mean_squared_error(y_test,ypred)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % np.sqrt(mse))
    RMSE.append(np.sqrt(mse))
    
    msg = "%s: %f (%f)" % (model, kf_cv_scores.mean(), kf_cv_scores.std())
    print(msg)
    print("============================================")
    print("============================================")
    print("============================================")
    


# # Model Evaluation

# In[100]:


Predictions.head()


# In[128]:


Predictions.sort_values('Production').tail(10)


# In[96]:


Predictions.plot(kind='scatter', x='Production', y='dtr_reg', color='r')
plt.title("Decision Tree")
Predictions.plot(kind='scatter', x='Production', y='knr_reg', color='g')
plt.title("kneighbors regression")
Predictions.plot(kind='scatter', x='Production', y='bagging_reg', color='b') 
plt.title("Bagging")
Predictions.plot(kind='scatter', x='Production', y='rf_reg', color='r')
plt.title("Random Forest")
Predictions.plot(kind='scatter', x='Production', y='grad_reg', color='g')
plt.title("Gradient Boosting")
plt.show()


# # Cross Validation Score

# In[92]:


for i in range(5):
    print(f"{models[i][0]}  : {results[i].mean()}")


# Gradient Boosting method has highest cross validation score, followed by Bagging regression

# In[93]:


fig = plt.figure()
fig.suptitle('Machine Learning Model Comparison')
ax = fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# # Root mean square error

# In[102]:


for name, model in models:
    MSE = np.square(np.subtract(Predictions.Production,Predictions[name])).mean()
    rmse =np.sqrt(MSE)
    print(f"{name}    : {rmse}")


# Bagging regression have the least root mean square error, Followed by Random forest and Decision tree respectively

# In[124]:


from sklearn.metrics import mean_squared_error, r2_score

for name, model in models:
    r2score=r2_score(Predictions.Production,Predictions[name])
    print(f"{name}    : {r2score}")


# In[ ]:




