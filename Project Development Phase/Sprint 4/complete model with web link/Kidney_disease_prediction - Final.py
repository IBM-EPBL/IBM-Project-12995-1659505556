#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this project we will predict the chances of getting a disease in Kidney.
# 
# The data was taken over a 2-month period in India with 25 features ( eg, red blood cell count, white blood cell count, etc). The target is the 'classification', which is either 'ckd' or 'notckd' - ckd=chronic kidney disease. There are 400 rows
# 
# ## Attribute Information:
# 
# We have 24 + class = 25 ( 11 numeric ,14 nominal) attributes/columns
# - **Age(numerical):** 
#     age in years
# - **Blood Pressure(numerical):** 
#     bp in mm/Hg
# - **Specific Gravity(nominal):** 
# sg - (1.005,1.010,1.015,1.020,1.025)
# - **Albumin(nominal):** 
# al - (0,1,2,3,4,5)
# - **Sugar(nominal):** 
# su - (0,1,2,3,4,5)
# - **Red Blood Cells(nominal):** 
# rbc - (normal,abnormal)
# - **Pus Cell (nominal):** 
# pc - (normal,abnormal)
# - **Pus Cell clumps(nominal):** 
# pcc - (present,notpresent)
# - **Bacteria(nominal):** 
# ba - (present,notpresent)
# - **Blood Glucose Random(numerical):** 
# bgr in mgs/dl
# - **Blood Urea(numerical):** 
# bu in mgs/dl
# - **Serum Creatinine(numerical):** 
# sc in mgs/dl
# - **Sodium(numerical):** 
# sod in mEq/L
# - **Potassium(numerical):** 
# pot in mEq/L
# - **Hemoglobin(numerical):** 
# hemo in gms
# - **Packed Cell Volume(numerical):** pcv
# - **White Blood Cell Count(numerical):** 
# wc in cells/cumm
# - **Red Blood Cell Count(numerical):** 
# rc in millions/cmm
# - **Hypertension(nominal):** 
# htn - (yes,no)
# - **Diabetes Mellitus(nominal):** 
# dm - (yes,no)
# - **Coronary Artery Disease(nominal):** 
# cad - (yes,no)
# - **Appetite(nominal):** 
# appet - (good,poor)
# - **Pedal Edema(nominal):** 
# pe - (yes,no)
# - **Anemia(nominal):** 
# ane - (yes,no)
# - **Class (nominal):** 
# class - (ckd,notckd) ckd - Chronic Kidney Disease, notckd - Not Chronic Kidney Disease
# 
# The dataset is available at: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease

# # Meaning of each Attributes
# 
# - **Specific Gravity (sg):** 
#     - Specific gravity is the ratio of weight of a given volume of a fluid (it can be Urine) to the weight of the same volume of distilled water measured at 25°C. 
#     - Measurement of specific gravity provides information regarding a patient's state of hydration or dehydration.
#     - Specific gravity is usually 1.010-1.025 (normal range: 1.003-1.030) and  highest in the morning. A value >1.025 indicates normal concentrating ability. 
# - **Albumin (al):**
#     - Albumin is a protein made by your liver.
#     - Albumin helps keep fluid in your bloodstream so it doesn't leak into other tissues. It is also carries various substances throughout your body, including hormones, vitamins, and enzymes.
#     -  Low albumin levels can indicate a problem with your liver or kidneys.
# - **Sugar (su):**
#     - Over time, the high levels of sugar in the blood damage the millions of tiny filtering units within each kidney. This eventually leads to kidney failure. 
#     - Around 20 to 30 per cent of people with diabetes develop kidney disease (diabetic nephropathy), although not all of these will progress to kidney failure.
# - **Red Blood Cell (rbc):**
#     - Red blood cells are responsible for transporting oxygen from your lungs to your body's tissues.
#     - When your kidneys are damaged, they produce less erythropoietin (EPO), a hormone that signals your bone marrow—the spongy tissue inside most of your bones—to make red blood cells. With less EPO, your body makes fewer red blood cells, and less oxygen is delivered to your organs and tissues.
# - **Pus Cell (pc):**
#     - They are neutrophils that have reached the site of infection as an immune response against infectious organisms (such as bacteria).
#     - Presence of pus cells in urine may indicate the presence of urinary tract infection (UTI). Presence of protein and red blood cells (RBCs) provides diagnostic clues for inflammatory kidney disease (i.e. glomerulonephritis).
# - **Bacteria (ba):**
#     - Bacteria causes Urinary Track Infection and hence there might be pus cell in urine.
# - **Blood Glucose Random (bgr):**
#     - It is a blood glucose levels at any given point in the day.
#     - Normal: 140 mg/dL or below
#     - Prediabetic: 140 – 199 mg/dL
#     - Diabetic: 200 mg/dL or above
# - **Blood Urea (bu):**
#     - Urea level in our blood
#     - Normal range: 6 to 24 mg/dL 
#     - But normal ranges may vary, depending on the reference range used by the lab and your age.
# - **Serum Creatinine (sc):**
#     - Amount of creatinine in your blood
#     - Creatinine is a waste product in your blood that comes from your muscles
#     - Normal range: For adult men, 0.74 to 1.35 mg/dL, for adult women, 0.59 to 1.04 mg/dL
# - **Sodium (sod):**
#     - Sodium help to conduct nerve impulses, contract and relax muscles, and maintain the proper balance of water and minerals.
#     - A normal blood sodium level is between 135 and 145 milliequivalents per liter (mEq/L).
# - **Potassium (pot):**
#     - It help in maintaining normal levels of fluid inside our cells. Sodium, its counterpart, maintains normal fluid levels outside of cells.
#     - The normal potassium level for an adult ranges from 3.5 to 5.2 mEq/L
# - **Hemoglobin (hemo):**
#     - Hemoglobin is a protein in your red blood cells that carries oxygen to your body's organs and tissues and transports carbon dioxide from your organs and tissues back to your lungs.
#     - The healthy range for hemoglobin is: For men, 13.2 (132 grams per liter) to 16.6 grams per deciliter. For women, 11.6 to 15 (116 grams per liter) grams per deciliter.
# - **Packed Cell Volume (pcv):**
#     - The packed cell volume (PCV) is a measurement of the proportion of blood that is made up of cells.
#     - In females, the normal range is 35.5 to 44.9%. In males, 38.3% to 48.6% is the normal PCV range. For pregnant females, the normal PCV is 33-38%.
# - **White blood cell count (wc):**
#     - A white blood count measures the number of white cells in your blood. 
#     - White blood cells are part of the immune system. 
#     - 3.8-9.9 WBC K/cumm is normal range
# - **Red Blood cell count (rc):**
#     - A red blood count measures the number of red cells in your blood.
#     - A normal range in adults is generally considered to be 4.35 to 5.65 million red blood cells per microliter (mcL) of blood for men and 3.92 to 5.13 million red blood cells per mcL of blood for women.
# - **Hypertension (htn):**
#     - High blood pressure (hypertension) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.
# - **Diabetes Mellitus (dm):**
#     - Diabetes mellitus refers to a group of diseases that affect how your body uses blood sugar (glucose).
# - **Coronary Artery Disease (cad):**
#     - Coronary artery disease is caused by plaque buildup in the wall of the arteries that supply blood to the heart (called coronary arteries).
# - **Appetite (appet):**
#     - Desire for eating food
# - **Pedal edema (pe):**
#     - Pedal edema causes an abnormal accumulation of fluid in the ankles, feet, and lower legs causing swelling of the feet and ankles.
# - **Anemia (ane):**
#     - Anemia is a condition in which you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues. 

# # Import necessary libraries and dataset

# In[1]:


# Importing necessary libraries used for data cleaning, and data visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

# Ignoring ununnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Importing library to split the data into training part and testing part.
from sklearn.model_selection import train_test_split

# Chi Square
from sklearn.feature_selection import chi2
import scipy.stats as stats

# Importing library to process the data (Normalize the data)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Importing Models (used for making prediction)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC                            # Support vector machine model
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore

# Importing metrics used for evaluation of our models
from sklearn import metrics
from sklearn.metrics import classification_report

# Hyperparameter tuner and Cross Validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# RandomOverSampler to handle imbalanced data
from imblearn.over_sampling import RandomOverSampler


# In[ ]:


sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pd.set_option('display.max_columns',29)


# # Data Collection

# In[3]:


df = pd.read_csv("kidney.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# **Observations:**
# - "classification" column is our target feature. "id" column has no use in prediction part so we will remove it later (before prediction).

# # Exploratory Data Analysis (EDA)

# In[6]:


# Checking the number of rows and columns in our dataset
df.shape


# - Dataset contains 400 rows and 26 columns

# In[7]:


# Getting more information of our dataset
df.info()


# **Observations:**
# - Seems there are some null values. We will remove it but before removing we will check null values by another technique.
# - Also dtype of many columns are wrong. For ex. pcv column should have dtype of int instead of object.

# In[8]:


df.isnull().sum()


# **Observations:**
# - There are many null values in our dataset. We will replace null values in Data Preprocessing part.

# In[9]:


# Getting some statistical information of our data
df.describe()


# - Count of each column is different which indicates presence of null values. We already seen that there is null values in our dataset
# - Our dataset contains age range from 2 to 90 years.

# In[10]:


# Distribution of our target variable i.e. "classification" column
df["classification"].value_counts()


# In[11]:


for i in df.drop("id",axis=1).columns:
    print('Unique Values in "{}":\n'.format(i),df[i].unique(), "\n\n")


# **Observations:**
# - Data cleaning much needed as there is \t after a word. 
# - For ex. Classification should be only ckd or notckd but there is one extra variable i.e. ckd\t. It should be actually ckd. Hence Data Cleaning much needed.

# - Columns from where we have to remove "\t" is: pcv, wc, rc, dm, cad, classification

# # Feature Engineering

# ## Replacing "\t"

# In[12]:


df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']] = df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']].replace(to_replace={'\t8400':'8400', '\t6200':'6200', '\t43':'43', '\t?':np.nan, '\tyes':'yes', '\tno':'no', 'ckd\t':'ckd', ' yes':'yes'})


# - Let's check still if there any "\t" in our data

# In[13]:


for i in df.drop("id",axis=1).columns:
    print('Unique Values in "{}":\n'.format(i),df[i].unique(), "\n\n")


# **Observations:**
# - Seems all "\t" removed from our data.
# - Now let's handle null values.

# ## Handling Null Values

# - Lets check null values in form of a graph first

# In[14]:


style.use('seaborn-darkgrid')

d = ((df.isnull().sum()/df.shape[0])).sort_values(ascending=False)
# Here we are plotting null values in range of 0-1. It means y axis range is 0-1.
# If bar graph show 0.5 null values that means there are 50% null values in that particular column.
# Hence we are dividing number of null values of each column with total number of rows i.e. 400 (or df.shape[0])

d.plot(kind = 'bar',
       color = sns.cubehelix_palette(start=2,
                                    rot=0.15,
                                    dark=0.15,
                                    light=0.95,
                                    reverse=True,
                                    n_colors=24),
        figsize=(20,10))
plt.title("\nProportions of Missing Values:\n",fontsize=40)
plt.show()


# **Observations:**
# - From this graph we can observe that rbc (Red Blood Cell) column have highest null values followed by rc (Red Blood Cell count), wc (White Blood Cell count), etc.

# ### Age

# In[20]:


sns.distplot(df.age)


# - As data seems little negative skewed, we will replace age null values with median

# In[22]:


df["age"] = df["age"].replace(np.NaN, df["age"].median())


# ### Blood Pressure (bp)

# In[26]:


df.bp.unique()


# In[51]:


df.bp.mode()[0]


# In[29]:


df.bp = df.bp.replace(np.NaN, df.bp.mode()[0])


# ### Specific Gravity (sg)

# In[31]:


df.sg.unique()


# In[50]:


df.sg.mode()[0]


# In[32]:


df.sg = df.sg.replace(np.NaN, df.sg.mode()[0])


# ### Aluminium (al)

# In[33]:


df.al.unique()


# In[49]:


df.al.mode()[0]


# In[34]:


df.al = df.al.replace(np.NaN, df.al.mode()[0])


# ### Sugar (su)

# In[35]:


df.su.unique()


# In[48]:


df.su.mode()[0]


# In[36]:


df.su = df.su.replace(np.NaN, df.su.mode()[0])


# ### Red blood cell (rbc)

# In[37]:


df.su.unique()


# In[47]:


df.rbc.mode()[0]


# In[38]:


df.rbc = df.rbc.replace(np.NaN, df.rbc.mode()[0])


# ### pc

# In[39]:


df.pc.unique()


# In[46]:


df.pc.mode()[0]


# In[41]:


df.pc = df.pc.replace(np.NaN, df.pc.mode()[0])


# ### pcc

# In[42]:


df.pcc.unique()


# In[45]:


df.pcc.mode()[0]


# In[43]:


df.pcc = df.pcc.replace(np.NaN, df.pcc.mode()[0])


# ### ba

# In[44]:


df.ba.unique()


# In[52]:


df.ba.mode()[0]


# In[53]:


df.ba = df.ba.replace(np.NaN, df.ba.mode()[0])


# ### bgr

# In[55]:


sns.distplot(df.bgr)


# - Seems positive skewed so we will replace nan with median

# In[56]:


df.bgr.median()


# In[57]:


df.bgr = df.bgr.replace(np.NaN, df.bgr.median())


# ### bu

# In[59]:


sns.distplot(df.bu)


# - Seems positive skewed so we will replace nan with median

# In[60]:


df.bu.median()


# In[61]:


df.bu = df.bu.replace(np.NaN, df.bu.median())


# ### sc

# In[63]:


sns.distplot(df.sc)


# - Seems positive skewed so we will replace nan with median

# In[64]:


df.sc.median()


# In[65]:


df.sc = df.sc.replace(np.NaN, df.sc.median())


# ### sod

# In[67]:


sns.distplot(df.sod)


# - Seems negative skewed so we will replace nan with median

# In[68]:


df.sod.median()


# In[69]:


df.sod = df.sod.replace(np.NaN, df.sod.median())


# ### pot

# In[71]:


sns.distplot(df.pot)


# - Seems positive skewed so we will replace nan with median

# In[72]:


df.pot.median()


# In[73]:


df.pot = df.pot.replace(np.NaN, df.pot.median())


# ### hemo

# In[75]:


sns.distplot(df.hemo)


# In[76]:


df.hemo.skew(skipna = True)


# - Seems little negative skewed so we will replace nan with median

# In[77]:


df.hemo.median()


# In[78]:


df.hemo = df.hemo.replace(np.NaN, df.hemo.median())


# ### pcv

# In[80]:


sns.distplot(df.pcv)


# In[82]:


df.pcv.skew(skipna = True)


# - Seems little negative skewed so we will replace nan with median

# In[83]:


df.pcv.median()


# In[84]:


df.pcv = df.pcv.replace(np.NaN, df.pcv.median())


# ### wc

# In[86]:


sns.distplot(df.wc)


# Seems positive skewed so we will replace nan with median

# In[87]:


df.wc.median()


# In[88]:


df.wc = df.wc.replace(np.NaN, df.wc.median())


# ### rc

# In[90]:


sns.distplot(df.rc)


# In[91]:


df.rc.skew(skipna = True)


# - Seems little negative skewed so we will replace nan with median

# In[92]:


df.rc.median()


# In[93]:


df.rc = df.rc.replace(np.NaN, df.rc.median())


# ### htn

# In[94]:


df.htn.unique()


# In[96]:


df.htn.mode()


# In[97]:


df.htn = df.htn.replace(np.NaN, df.htn.mode()[0])


# ### dm

# In[98]:


df.dm.mode()


# In[99]:


df.dm = df.dm.replace(np.NaN, df.dm.mode()[0])


# ### cad

# In[100]:


df.cad.unique()


# In[101]:


df.cad.mode()


# In[102]:


df.cad = df.cad.replace(np.NaN, df.cad.mode()[0])


# ### appet

# In[103]:


df.appet.unique()


# In[104]:


df.appet.mode()


# In[105]:


df.appet = df.appet.replace(np.NaN, df.appet.mode()[0])


# ### pe

# In[106]:


df.pe.unique()


# In[107]:


df.pe.mode()


# In[108]:


df.pe = df.pe.replace(np.NaN, df.pe.mode()[0])


# ### ane

# In[110]:


df.ane.unique()


# In[111]:


df.ane.mode()


# In[112]:


df.ane = df.ane.replace(np.NaN, df.ane.mode()[0])


# In[113]:


df.isnull().sum()


# - All null values are replaced by using forward fill method and backward fill method.

# ## Handling Dtypes

# In[114]:


df.dtypes


# In[115]:


for i in df.columns:
    print('Unique Values in "{}":\n'.format(i), df[i].unique(), "\n-----------------------------------------------------\n")


# **Observations:**
# - pcv, wc, and rc column should have dtype of float as it contains all integer but it is of object type. We have to correct this.

# In[116]:


df['rc'] = df['rc'].astype('float64')
df[['pcv', 'wc', 'age']] = df[['pcv', 'wc', 'age']].astype('int64')
df.dtypes


# In[117]:


display(df['pcv'].unique())
display(df['wc'].unique())
display(df['rc'].unique())


# - Yipee! We have changed dtypes successfully.

# ## Dropping unnecessary columns

# - id column is not required for our data analysis as it is not providing any info which will be useful in making prediction.

# In[118]:


df.drop('id',axis=1,inplace=True)
df.head()


# # Data Visualization

# - Seperating numerical columns and categorical columns for easy data visualization code implementation

# In[119]:


sns.countplot(x = "classification", data = df)


# In[120]:


pp = sns.pairplot(df[["bp","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc","classification"]], hue = "classification", height=1.8, aspect=1.8, plot_kws=dict(edgecolor="k", linewidth=0.5), diag_kind="kde", diag_kws=dict(shade=True))
fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Chronic Kidney Disease', fontsize=30)


# **Observations:**
# - From pairplot, we can observe that maximum plot has some kind of linear relation, while some plot have non linear relation. Hence it is better to use Spearman correlation to find correlation percentage among attributes.

# #### Spearman

# In[121]:


sns.set(font_scale=0.45)
plt.title('Chronic Kidney Disease Attributes Correlation')
cmap = sns.diverging_palette(260, 10, as_cmap=True)
sns.heatmap(df[["age","bp","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]].corr("spearman"), vmax=1.2, annot=True, square='square', cmap=cmap, fmt = '.0%', linewidths=2)


# In[122]:


# With the following function we can select highly correlated features
# It will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr("spearman")
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[123]:


corr_features = correlation(df, 0.85)
corr_features


# In[124]:


sns.scatterplot(x="pcv", y="hemo", data=df)


# **Observations:**
# - From Heatmap and Scatterplot, we can easily observe that PCV and Hemoglobin is highly correlated with 88%. So we can remove anyone of this column as it is acting like duplicate of another.

# - From Heatmap and Scatterplot, we can observe that RBC count and PCV are 76% correlated
# - Also RBC count and hemoglobin are 75% correlated while Blood Urea and Serum Creatinine are 69% correlated.

# # Label Encoding

# - Label encoding will convert all categorical column with object as dtype to int64 as dtype. Example: Yes or No to 1 or 0.

# In[125]:


df.head()


# In[126]:


col = ['rbc', 'pcc', 'pc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane']
encoder = LabelEncoder()
for col in col:
    df[col] = encoder.fit_transform(df[col])


# In[127]:


df[['appet', 'classification']] = df[['appet', 'classification']].replace(to_replace={'good':'1', 'ckd':'1', 'notckd':'0', 'poor':'0'})


# In[128]:


df.head(2)


# In[129]:


df.dtypes


# In[130]:


df[['classification', 'appet']] = df[['classification', 'appet']].astype('int64')


# **Observations:**
# - We converted all object categorical to int categorical, it will surely increase the performance of our model.
# - Here, present = 1, normal = 1, yes = 1, notpresent = 0, abnormal = 0, no = 0, good = 1 , poor = 0, ckd = 1, notckd = 0.

# ## Feature Selection
# ### Relation among numerical and classification column

# In[131]:


df_anova = df[["age","bp","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc","classification"]]
grps = pd.unique(df_anova.classification.values)
grps

for i in range(len(df_anova.columns)-1):
    
    d_data = {grp:df_anova[df_anova.columns[i]][df_anova.classification == grp] for grp in grps}

    F, p = stats.f_oneway(d_data[0], d_data[1])
    print("P_Value of {} and Classification".format(df_anova.columns[i]), p)

    if p < 0.05:
        print("There is relation between {} and Classification \n".format(df_anova.columns[i]))
    else:
        print("There is no relation between {} and Classification \n".format(df_anova.columns[i]))


# #### Lets verify correlation between Potassium and Classification using Point biserial Correlation
# https://medium.com/@outside2SDs/an-overview-of-correlation-measures-between-categorical-and-continuous-variables-4c7f85610365#:~:text=There%20are%20three%20big%2Dpicture,and%20Kruskal%20Wallis%20H%20Test.&text=The%20point%20biserial%20correlation%20coefficient,case%20of%20Pearson's%20correlation%20coefficient.

# In[132]:


x = np.array(df.pot)
y = np.array(df.classification)
_, p = stats.pointbiserialr(x, y)
print(p)

if p < 0.05:
    print("There is relation between Potassium and Classification \n")
else:
    print("There is no relation between Potassium and Classification \n")


# **Observations:**
# - So we have to drop PCV and Potassium column
# - Dropping PCV because it has highly correlation with Hemoglobin.
# - Dropping Potassium because it has no correlation with Classification

# In[133]:


df.drop(["pcv","pot"], axis=1, inplace=True)
display(df.head())
df.shape


# # Seperating target feature

# In[134]:


X = df.drop("classification", axis=1)
y = df["classification"]
display(X)
display(y)


# ## Relation between categorical column

# In[138]:


f_p_values=chi2(X[['sg', 'al', 'su', 'rbc', 'pcc', 'pc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane', 'appet']],y)

p_values = pd.Series(f_p_values[1])
p_values.index = ['sg', 'al', 'su', 'rbc', 'pcc', 'pc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane', 'appet']
p_values.sort_values(ascending=False)


# In[139]:


# Null Hypothesis: The null hypothesis states that there is no relationship between the two variables
cnt = 0
for i in p_values:
    if i > 0.05:
        print("There is no relationship", p_values.index[cnt], i)
    else:
        print("There is relationship", p_values.index[cnt], i)
    
    cnt += 1


# In[137]:


p_values.index


# **Observations:**
# **Attributes which are correlated with depending label (classification)**
# - Specific Gravity has no correlation with classification
# - Similarly, Aluminium, sugar, RBC, pus cell clumps, pus cell, bacteria, hypertension, diabetes, coronary artery disease, anemia, appetide, and pedal edema has relation with classification

# In[140]:


df.drop("sg", axis=1, inplace=True)
display(df.head(2))
df.shape


# ### Dropping constant feature

# In[141]:


from sklearn.feature_selection import VarianceThreshold

var_thres = VarianceThreshold(threshold=0)
var_thres.fit(df)

var_thres.get_support()

print(df.columns[var_thres.get_support()])


constant_columns = [column for column in df.columns
                    if column not in df.columns[var_thres.get_support()]]
print(constant_columns)
print(len(constant_columns))
print("Shape: ", df.shape)


# - There is no constant feature in our dataset

# # Standardization, Splitting, and Balancing data

# ## Seperating target

# In[144]:


X = df.drop("classification", axis=1)
y = df["classification"]


# ## Standardization of the data

# In[145]:


scaler = StandardScaler()
features = scaler.fit_transform(X)
features


# - We will also standardize the data by using StandardScaler. This will help in increasing the performance of the model and increasing accuracy.

# ## Splitting the data into train and test data

# In[146]:


X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42) 


# ## Balancing Data

# In[147]:


len(y_train[y_train==1]), len(y_train[y_train==0]), y_train.shape


# - Data is somewhat imbalanced
# - We will do oversampling to balance it

# ### Under Sampling

# In[151]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()

X_train_down,y_train_down = rus.fit_resample(X_train, y_train)

print(len(y_train_down[y_train_down==0]), len(y_train_down[y_train_down==1]))
print(len(X_train_down))


# ### Over Sampling

# In[152]:


os =  RandomOverSampler(sampling_strategy=1)

X_train, y_train = os.fit_resample(X_train, y_train)

print(len(y_train[y_train==0]), len(y_train[y_train==1]))
print(len(X_train))


# # Model Building

# ## Logistic Regression

# In[153]:


def lr_grid_search(X, y):
    model = LogisticRegression()
    
    # Create a dictionary of all values we want to test
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    
    # define grid search
    param_grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
    grid_result = grid_search.fit(X, y)
    
    return grid_result.best_params_


# In[154]:


lr_grid_search(X_train, y_train)


# ### Over sample Logistic

# In[156]:


lr = LogisticRegression(C=1, penalty='l2', solver='newton-cg')
lr.fit(X_train,y_train)

y_pred_lr = lr.predict(X_test)

print(metrics.classification_report(y_test, y_pred_lr))

lr_score = lr.score(X_train,y_train)
print(lr_score)

lr_score = lr.score(X_test,y_test)
print(lr_score)


# - **Accuracy** = Proportion of correct prediction over total prediction
# - **Recall** = Out of actual positive(True positive + False negative), how many are True positive
# - **Precision** = Out of predicted positive (true positive + False positive), how many are True positive

# In[157]:


lr_tacc = lr.score(X_test,y_test)
lr_train_acc = lr.score(X_train, y_train)


# #### Confusion matrix of Logistic Regression Model

# In[160]:


cm = metrics.confusion_matrix(y_test, y_pred_lr, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# #### AUC of Logistic Regression Model

# In[161]:


y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[162]:


lr_auc = auc
lr_auc


# # Under Sample Logistic

# In[155]:


lr_grid_search(X_train_down, y_train_down)


# In[163]:


lr = LogisticRegression(C=10, penalty='l2', solver='newton-cg')
lr.fit(X_train_down,y_train_down)

y_pred_lr = lr.predict(X_test)

print(metrics.classification_report(y_test, y_pred_lr))

lr_score = lr.score(X_train_down,y_train_down)
print(lr_score)

lr_score = lr.score(X_test,y_test)
print(lr_score)


# In[164]:


cm = metrics.confusion_matrix(y_test, y_pred_lr, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# In[165]:


y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[166]:


lr_tacc_down = lr.score(X_test,y_test)
lr_train_acc_down = lr.score(X_train_down, y_train_down)
lr_auc_down = auc
lr_auc_down


# ## Decision Tree Classifier

# In[167]:


def dtree_grid_search(X, y):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(2, 15)}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # decision tree model
    dtree = DecisionTreeClassifier()
    
    #use gridsearch to test all values
    dtree_gscv = GridSearchCV(dtree, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    #fit model to data
    dtree_gscv.fit(X, y)
    
    return dtree_gscv.best_params_


# In[168]:


dtree_grid_search(X_train, y_train)


# ### Over Sample Decision Tree

# In[170]:


dTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11)
dTree.fit(X_train, y_train)

print(dTree.score(X_train,y_train))
print(dTree.score(X_test,y_test))

y_pred_dtree = dTree.predict(X_test)

print(metrics.classification_report(y_test, y_pred_dtree))


# In[171]:


dt_tacc = dTree.score(X_test,y_test)
dt_train_acc = dTree.score(X_train, y_train)


# #### Confusion Matrix of Decision Tree Classifier

# In[172]:


cm = metrics.confusion_matrix(y_test, y_pred_dtree, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# #### AUC of DecisionTree Model

# In[173]:


y_pred_proba = dTree.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[174]:


dt_auc = auc
dt_auc


# ### Under Sample Decision Tree

# In[169]:


dtree_grid_search(X_train_down, y_train_down)


# In[175]:


dTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 6)
dTree.fit(X_train_down, y_train_down)

print(dTree.score(X_train_down,y_train_down))
print(dTree.score(X_test,y_test))

y_pred_dtree = dTree.predict(X_test)

print(metrics.classification_report(y_test, y_pred_dtree))


# In[176]:


cm = metrics.confusion_matrix(y_test, y_pred_dtree, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# In[177]:


y_pred_proba = dTree.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[178]:


dt_tacc_down = dTree.score(X_test,y_test)
dt_train_acc_down = dTree.score(X_train_down, y_train_down)
dt_auc_down = auc
dt_auc_down


# ## Ensemble learning - AdaBoosting

# In[179]:


def ada_grid_search(X, y):
    #create a dictionary of all values we want to test
    param_grid = {'n_estimators':[10, 50, 100, 500], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # AdaBoost model
    ada = AdaBoostClassifier()
    
    # Use gridsearch to test all values
    ada_gscv = GridSearchCV(ada, param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
    #fit model to data
    grid_result = ada_gscv.fit(X, y)
    
    return ada_gscv.best_params_


# In[180]:


ada_grid_search(X_train, y_train)


# ### Over Sample AdaBoost

# In[182]:


abcl = AdaBoostClassifier(n_estimators=500, learning_rate = 0.1)
abcl = abcl.fit(X_train, y_train)

y_pred_abcl = abcl.predict(X_test)

print(abcl.score(X_train, y_train))
print(abcl.score(X_test,y_test))

print(metrics.classification_report(y_test, y_pred_abcl))


# In[183]:


ada_train_acc = abcl.score(X_train, y_train)
ada_tacc = abcl.score(X_test,y_test)


# #### Confusion Matrix AdaBoosting model

# In[184]:


cm = metrics.confusion_matrix(y_test, y_pred_abcl, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# #### AUC of Adaboosting model

# In[185]:


y_pred_proba = abcl.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[186]:


ada_auc = auc
ada_auc


# ### Under Sample AdaBoost

# In[181]:


ada_grid_search(X_train_down, y_train_down)


# In[187]:


abcl = AdaBoostClassifier(n_estimators=100, learning_rate = 0.1)
abcl = abcl.fit(X_train_down, y_train_down)

y_pred_abcl = abcl.predict(X_test)

print(abcl.score(X_train_down, y_train_down))
print(abcl.score(X_test,y_test))

print(metrics.classification_report(y_test, y_pred_abcl))


# In[188]:


cm = metrics.confusion_matrix(y_test, y_pred_abcl, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# In[189]:


y_pred_proba = abcl.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[205]:


ada_train_acc_down = abcl.score(X_train_down, y_train_down)
ada_tacc_down = abcl.score(X_test,y_test)
ada_auc_down = auc
ada_auc_down


# ## Random forest classifier

# In[191]:


def rf_grid_search(X, y):
    #create a dictionary of all values we want to test
    param_grid = { 
    'n_estimators': [5,10,20,40,50,60,70,80,100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
    }
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Random Forest model
    rf = RandomForestClassifier()
    
    #use gridsearch to test all values
    rf_gscv = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    #fit model to data
    rf_gscv.fit(X, y)
    
    return rf_gscv.best_params_


# In[192]:


rf_grid_search(X_train, y_train)


# ### Over Sample Random Forest

# In[194]:


rfcl = RandomForestClassifier(n_estimators=70, max_features='sqrt', max_depth=7, criterion='entropy')
rfcl = rfcl.fit(X_train, y_train)

y_pred_rf = rfcl.predict(X_test)

print(rfcl.score(X_train,y_train))
print(rfcl.score(X_test,y_test))

print(metrics.classification_report(y_test, y_pred_rf))


# In[195]:


rf_tacc = rfcl.score(X_test,y_test)
rf_train_acc = rfcl.score(X_train, y_train)


# #### Confusion matrix of Random Forest Classifier Model

# In[196]:


cm = metrics.confusion_matrix(y_test, y_pred_rf, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# #### AUC of Random Forest Classifier Model

# In[197]:


y_pred_proba = rfcl.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[198]:


rf_auc = auc
rf_auc


# ### Under Sample Random Forest

# In[193]:


rf_grid_search(X_train_down, y_train_down)


# In[200]:


rfcl = RandomForestClassifier(n_estimators=80, max_features='log2', max_depth=7, criterion='entropy')
rfcl = rfcl.fit(X_train_down, y_train_down)

y_pred_rf = rfcl.predict(X_test)

print(rfcl.score(X_train_down,y_train_down))
print(rfcl.score(X_test,y_test))

print(metrics.classification_report(y_test, y_pred_rf))


# In[201]:


cm = metrics.confusion_matrix(y_test, y_pred_rf, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# In[202]:


y_pred_proba = rfcl.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[204]:


rf_tacc_down = rfcl.score(X_test,y_test)
rf_train_acc_down = rfcl.score(X_train_down, y_train_down)
rf_auc_down = auc
rf_auc_down


# ## kNN

# In[206]:


def knn_grid_search(X, y):
    #create a dictionary of all values we want to test
    k_range = list(range(1, 31))
    param_grid = dict(n_neighbors=k_range)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    knn = KNeighborsClassifier()
    
    #use gridsearch to test all values
    knn_gscv = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    #fit model to data
    knn_gscv.fit(X, y)
    
    return knn_gscv.best_params_


# In[207]:


knn_grid_search(X_train, y_train)


# ### Over Sample kNN

# In[209]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))

print(metrics.classification_report(y_test, y_pred_knn))


# In[210]:


knn_tacc = knn.score(X_test, y_test)
knn_train_acc = knn.score(X_train, y_train)


# #### Confusion Matrix of kNN

# In[211]:


cm = metrics.confusion_matrix(y_test, y_pred_knn, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# #### AUC of kNN

# In[212]:


y_pred_proba = knn.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[213]:


knn_auc= auc
knn_auc


# ### Under Sample kNN

# In[208]:


knn_grid_search(X_train_down, y_train_down)


# In[214]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_down, y_train_down)

y_pred_knn = knn.predict(X_test)

print(knn.score(X_train_down, y_train_down))
print(knn.score(X_test, y_test))

print(metrics.classification_report(y_test, y_pred_knn))


# In[215]:


cm = metrics.confusion_matrix(y_test, y_pred_knn, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# In[216]:


y_pred_proba = knn.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[217]:


knn_tacc_down = knn.score(X_test, y_test)
knn_train_acc_down = knn.score(X_train_down, y_train_down)
knn_auc_down = auc
knn_auc_down


# ## SVM

# In[219]:


def svm_grid_search(X, y):
    #create a dictionary of all values we want to test
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001, 0.4, 0.2, 0.8],'kernel': ['rbf', 'poly', 'sigmoid']}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    svm = SVC()
    
    #use gridsearch to test all values
    svm_gscv = RandomizedSearchCV(estimator = svm,
                           param_distributions = param_grid,
                           scoring = 'accuracy',
                           cv = cv,
                           n_jobs = -1)
    #fit model to data
    svm_gscv.fit(X, y)
    
    return svm_gscv.best_params_


# In[220]:


svm_grid_search(X_train, y_train)


# ### Over Sample SVM

# In[242]:


from sklearn import svm
svm = SVC(gamma=0.8, C=10, kernel='rbf', probability=True)

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))

print(metrics.classification_report(y_test, y_pred_svm))


# In[243]:


svm_tacc = svm.score(X_test, y_test)
svm_train_acc = svm.score(X_train, y_train)


# #### Confusion Matrix of SVM

# In[244]:


cm = metrics.confusion_matrix(y_test, y_pred_svm, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# #### AUC of SVM

# In[245]:


y_pred_proba = svm.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[246]:


svm_auc = auc
svm_auc


# ### Under Sample SVM

# In[221]:


svm_grid_search(X_train_down, y_train_down)


# In[247]:


from sklearn import svm
svm = SVC(gamma=0.4, C=1, kernel='rbf', probability=True)

svm.fit(X_train_down, y_train_down)

y_pred_svm = svm.predict(X_test)

print(svm.score(X_train_down, y_train_down))
print(svm.score(X_test, y_test))

print(metrics.classification_report(y_test, y_pred_svm))


# In[248]:


cm = metrics.confusion_matrix(y_test, y_pred_svm, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# In[249]:


y_pred_proba = svm.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[250]:


svm_tacc_down = svm.score(X_test, y_test)
svm_train_acc_down = svm.score(X_train_down, y_train_down)
svm_auc_down = auc
svm_auc_down


# ## XGBoost Model

# In[231]:


def xgb_grid_search(X, y):
    # Create a dictionary of all values we want to test
    param_grid = {
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    xgb = XGBClassifier()
    
    #use gridsearch to test all values
    xgb_gscv =  RandomizedSearchCV(estimator = xgb,
                           param_distributions = param_grid,
                           scoring = 'accuracy',
                           cv = cv,
                           n_jobs = -1)
    #fit model to data
    xgb_gscv.fit(X, y)
    
    return xgb_gscv.best_params_


# In[232]:


xgb_grid_search(X_train, y_train)


# ### Over Sample XGBoost

# In[234]:


xgb = XGBClassifier(min_child_weight=1, max_depth=10, learning_rate=0.25, gamma=0.4, colsample_bytree=0.3)
xgb.fit(X_train,y_train)

y_pred_xgb = xgb.predict(X_test)

print(classification_report(y_test, y_pred_xgb))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_xgb))
print("Precision:",metrics.precision_score(y_test, y_pred_xgb))
print("Recall:",metrics.recall_score(y_test, y_pred_xgb))

print(xgb.score(X_train,y_train))
print(xgb.score(X_test,y_test))


# In[235]:


xgb_tacc = xgb.score(X_test,y_test)
xgb_train_acc = xgb.score(X_train, y_train)


# #### Confusion Matrix of XGBoost

# In[236]:


cm = metrics.confusion_matrix(y_test, y_pred_xgb, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# #### AUC of XGBoost Model

# In[237]:


y_pred_proba = xgb.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[238]:


xgb_auc = auc


# ### Under Sample XGBoost

# In[233]:


xgb_grid_search(X_train_down, y_train_down)


# In[251]:


xgb = XGBClassifier(min_child_weight=1, max_depth=8, learning_rate=0.15, gamma=0.1, colsample_bytree=0.7)
xgb.fit(X_train_down,y_train_down)

y_pred_xgb = xgb.predict(X_test)

print(classification_report(y_test, y_pred_xgb))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_xgb))
print("Precision:",metrics.precision_score(y_test, y_pred_xgb))
print("Recall:",metrics.recall_score(y_test, y_pred_xgb))

print(xgb.score(X_train_down,y_train_down))
print(xgb.score(X_test,y_test))


# In[252]:


cm = metrics.confusion_matrix(y_test, y_pred_xgb, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# In[253]:


y_pred_proba = xgb.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[254]:


xgb_tacc_down = xgb.score(X_test,y_test)
xgb_train_acc_down = xgb.score(X_train_down, y_train_down)
xgb_auc_down = auc


# # Comparision of all Models

# ## Over Sample Models

# In[255]:


def comp_model(model_list, model_train_acc_list, model_test_acc_list, model_auc_list):
    data = {"Model Name": model_list, "Train Accuracy(%)": [i*100 for i in model_train_acc_list], "Test Accuracy(%)": [i*100 for i in model_test_acc_list], "AUC Score": model_auc_list}
    Comparision = pd.DataFrame(data)
    return Comparision


# In[256]:


model_list = ["Logistic Regression", "Decision Tree Classifier", "AdaBoost", "Random Forest Classifier", "kNN", "SVM", "XGBoost"]
model_train_acc_list = [lr_train_acc, dt_train_acc, ada_train_acc, rf_train_acc, knn_train_acc, svm_train_acc, xgb_train_acc]
model_test_acc_list = [lr_tacc, dt_tacc, ada_tacc, rf_tacc, knn_tacc, svm_tacc, xgb_tacc]
model_auc_list = [lr_auc, dt_auc, ada_auc, rf_auc, knn_auc, svm_auc, xgb_auc]
comp_model(model_list, model_train_acc_list, model_test_acc_list, model_auc_list)


# - The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes
# - We can say that Random Forest Classifier Model and AdaBoost Model are good for our over sampled dataset as it is giving highest AUC score as well as highest accuracy.
# - Lets do Cross Validation and find out which model is Best

# ## Under Sample Models

# In[257]:


model_list = ["Logistic Regression", "Decision Tree Classifier", "AdaBoost", "Random Forest Classifier", "kNN", "SVM", "XGBoost"]
model_train_acc_list = [lr_train_acc_down, dt_train_acc_down, ada_train_acc_down, rf_train_acc_down, knn_train_acc_down, svm_train_acc_down, xgb_train_acc_down]
model_test_acc_list = [lr_tacc_down, dt_tacc_down, ada_tacc_down, rf_tacc_down, knn_tacc_down, svm_tacc_down, xgb_tacc_down]
model_auc_list = [lr_auc_down, dt_auc_down, ada_auc_down, rf_auc_down, knn_auc_down, svm_auc_down, xgb_auc_down]
comp_model(model_list, model_train_acc_list, model_test_acc_list, model_auc_list)


# - The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes
# - We can say that Decision Tree Classifier Model, Random Forest Classifier Model, and XGBoost Model are good for our under sampled dataset as it is giving highest AUC score as well as highest accuracy.
# - Lets do Cross Validation and find out which model is Best

# # Cross Validation

# ## Without over or down sampling

# ### Random Forest Classfier

# In[269]:


skfold = StratifiedKFold(n_splits=5)
model = RandomForestClassifier(n_estimators=70, max_features='sqrt', max_depth=7, criterion='entropy')
scores = cross_val_score(model, features, y, cv=skfold)

print(scores)
print(np.mean(scores))


# ### AdaBoost Classifier

# In[270]:


skfold = StratifiedKFold(n_splits=5)
model = AdaBoostClassifier(n_estimators=500, learning_rate = 0.1)
scores = cross_val_score(model, features, y, cv=skfold)

print(scores)
print(np.mean(scores))


# ### XGBoost Classifier

# In[271]:


skfold = StratifiedKFold(n_splits=5)
model = XGBClassifier(min_child_weight=1, max_depth=10, learning_rate=0.25, gamma=0.4, colsample_bytree=0.3)
scores = cross_val_score(model, features, y, cv=skfold)

print(scores)
print(np.mean(scores))


# ## Over Sample CV

# In[261]:


len(y[y==0]), len(y[y==1])


# In[259]:


os =  RandomOverSampler(sampling_strategy=1)

X_train, y_train = os.fit_resample(features, y)

print(len(y_train[y_train==0]), len(y_train[y_train==1]))
print(len(X_train))


# ### Random Forest Classifier

# In[262]:


skfold = StratifiedKFold(n_splits=5)
model = RandomForestClassifier(n_estimators=70, max_features='sqrt', max_depth=7, criterion='entropy')
scores = cross_val_score(model, X_train, y_train, cv=skfold)

print(scores)
print(np.mean(scores))


# ### AdaBoost Model

# In[263]:


skfold = StratifiedKFold(n_splits=5)
model = AdaBoostClassifier(n_estimators=500, learning_rate = 0.1)
scores = cross_val_score(model, X_train, y_train, cv=skfold)

print(scores)
print(np.mean(scores))


# ### XGBoost Model

# In[268]:


skfold = StratifiedKFold(n_splits=5)
model = XGBClassifier(min_child_weight=1, max_depth=10, learning_rate=0.25, gamma=0.4, colsample_bytree=0.3)
scores = cross_val_score(model, X_train, y_train, cv=skfold)

print(scores)
print(np.mean(scores))


# ## Under Sample CV

# In[264]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()

X_train_down,y_train_down = rus.fit_resample(features, y)

print(len(y_train_down[y_train_down==0]), len(y_train_down[y_train_down==1]))
print(len(X_train_down))


# ### XGBoost Classifier

# In[265]:


skfold = StratifiedKFold(n_splits=5)
model = XGBClassifier(min_child_weight=1, max_depth=8, learning_rate=0.15, gamma=0.1, colsample_bytree=0.7)
scores = cross_val_score(model, X_train_down,y_train_down, cv=skfold)

print(scores)
print(np.mean(scores))


# ### Decision Tree Classifier

# In[266]:


skfold = StratifiedKFold(n_splits=5)
model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 6)
scores = cross_val_score(model, X_train_down,y_train_down, cv=skfold)

print(scores)
print(np.mean(scores))


# ### Random Forest Classifier

# In[267]:


skfold = StratifiedKFold(n_splits=5)
model = RandomForestClassifier(n_estimators=80, max_features='log2', max_depth=7, criterion='entropy')
scores = cross_val_score(model, X_train_down, y_train_down, cv=skfold)

print(scores)
print(np.mean(scores))


# **Observations:**
# - From this Cross Validation, we can conclude that **Random Forest Classifier model** is best for our project. Also model showing hgiher accuracy and auc score in **over sampling**.

# # Building the Prediction System - Random Forest Classifier

# In[272]:


#input data and transform into numpy array
in_data= np.asarray(tuple(map(float,input().rstrip().split(','))))

#reshape and scale the input array
in_data_re = in_data.reshape(1,-1)
in_data_sca = scaler.transform(in_data_re)

#print the predicted output for input array
print("Chronic Kidney Disease Detected" if rfcl.predict(in_data_sca) else "Chronic Kidney Disease Not Detected")


# ***Extra data on which you can try our both Prediction System***
# 
# Chronic Disease Positive:
# - [48, 80, 1, 0, 1, 1, 0, 0, 121, 36, 1.2, 111, 15.4, 7800, 5.2, 1, 1, 0, 1, 0, 0]
# - [7, 50, 4, 0, 1, 1, 0, 0, 121, 18, 0.8, 111, 11.3, 6000, 5.2, 0, 0, 0, 1, 0, 0]
# 
# Chronic Disease Negative:
# - [40, 80, 0, 0, 1, 1, 0, 0, 140, 10, 1.2, 135, 15, 10400, 4.5, 0, 0, 0, 1, 0, 0]
# - [23, 80, 0, 0, 1, 1, 0, 0, 70, 36, 1, 150, 17, 9800, 5, 0, 0, 0, 1, 0, 0]

# # Multilayer Neural Network

# In[273]:


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from keras.layers import Dense, Activation, LeakyReLU, Dropout
from keras.activations import relu, sigmoid

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve


# ## Hyperparameter Tuning

# **Hyperparameters**
# - How many number of hidden layers we should have?
# - How many number of neurons we should have in hidden layers?
# - Learning Rate

# In[274]:


X_train.shape


# In[275]:


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units_inp', min_value=32, max_value=512, step=32),
                               activation=hp.Choice( 'activation', ['tanh', 'relu', 'LeakyReLU', 'elu']),
                               input_dim = 21
                          )
              )
    
    for i in range(hp.Int('num_layers', 2, 20)):                 # Number of layers
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,       # Number of neuron, here it is 32-512
                                            step=32),
                               activation=hp.Choice( 'activation', ['tanh', 'relu', 'LeakyReLU', 'elu'])))
                               
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))
        
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    
    return model


# In[276]:


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='MultiNN',
    project_name='Kidney Disease Detection')


# In[277]:


tuner.search_space_summary()


# In[278]:


tuner.search(X_train, y_train, epochs=5, validation_split=0.2)


# In[279]:


tuner.results_summary()


# In[284]:


best_model = tuner.get_best_models(num_models=1)[0]


# In[285]:


best_model.summary()


# In[286]:


best_model.fit(X_train,y_train,epochs=50)


# In[287]:


# Train and Test accuracy
scores = best_model.evaluate(X_train,y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = best_model.evaluate(X_test,y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))


# In[288]:


y_test_pred_probs = best_model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test,y_test_pred_probs)

auc = metrics.roc_auc_score(y_test, y_test_pred_probs)
auc

plt.plot(FPR,TPR,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.plot([0,1],[0,1],'--',color='black')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# **Conclusion:**
# - Using Dense Neural Network, we are getting almost 100% training accuracy and 100% test accuracy which seems to be very good.
# - Hence both Dense Neural Network and Random Forest Model is best for this project to predict whether a patient has CKD or not
