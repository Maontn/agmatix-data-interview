
# coding: utf-8

# # Intro

# In[1]:


# Since purpose and design (including tools, best practices and future suggestions)
# had already been presented and writtend in first interview/test on premises
# , then jumping into actual code w/ clarifications


# In[2]:


# So, this code is to be invoked in the env where source files are located
# digesting and cleaning the data
# uploading to db


# # Imports

# In[3]:


import pandas as pd
import os
import re
from datetime import datetime
from IPython.display import display
import numpy as np


# In[4]:


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# # Start logging report

# In[5]:


log_df_cols = ['timestamp', 'severity', 'flag', 'msg']

log_df = pd.DataFrame(data=[[datetime.now(), 'high', 'success','started log_df']]
                      ,columns=log_df_cols)

def add_log(log_vals):
    global log_df
    log_vals = [datetime.now()]+log_vals
    log_df = pd.concat([pd.DataFrame(data=[log_vals], columns=log_df_cols), log_df])
    log_df.head()
    return log_df

log_df


# # Load all files in desitination folder

# In[6]:


# folder_path  = input('enter folder_path')
folder_path  = r'C:\Users\Noam\Desktop\Agmatix\agmatix-data-interview\agriculture'
# C:\Users\Noam\Desktop\Agmatix\agmatix-data-interview\agriculture


# In[7]:


file_names = os.listdir(folder_path)
file_names = [f for f in file_names if (f[-3:] == 'xls' or f[-4:] == 'xlsx') and f[0] != '~']
print (file_names)


# In[8]:


add_log(['high', 'success','folder path entered and f_names parsed, total: '+str(len(file_names))])


# ## convert files to Dataframe
# ## w/ files integrity check : readable and shape

# In[9]:


df_cnt = 0
src_list = []
for f in file_names:
    df_name = f.split('.')[0]+'_df'
    path = folder_path+'\\'+f
    path = path.replace('\\','/') 
    
    file = pd.ExcelFile(path)
    sheets = file.sheet_names

    for i in range(len(sheets)):
        s = sheets[i]
        df_s_name = df_name+'_'+s
        df_s_name = df_s_name.replace(' ','').replace('-','_')
        try:
            cmd  = r'''{df_s_name} = pd.read_excel('{path}', sheet_name={i})'''.format(df_s_name=df_s_name,path=path, i=i)
            exec(cmd)
            df_cnt += 1
            exec(r'''df_shp = {df_s_name}.shape'''.format(df_s_name=df_s_name))
            assert df_shp!=(0,0), r'''{df_s_name} is empty'''.format(df_s_name=df_s_name)
            add_log(['high', 'success','loaded file to dataframe: '+df_s_name+' shape: '+str(df_shp)])
            src_list.append(df_s_name)
            print(df_s_name)
            exec(r'''display({df_s_name}.head(2))'''.format(df_s_name=df_s_name))
        except:
            add_log(['high', 'fail','fail to load file to dataframe: '+df_s_name])
display(log_df)


# In[10]:


# delete irrelevant sources according to assignment info

src_del = ['sensors_df_50cm', 'sensors_df_75cm']
src_list = [s for s in src_list if s not in src_del]
print(src_list)

add_log(['low', 'success','delete irrelevant sources'])


# # Reformat sets & Data validaiton

# ### Null records

# In[11]:


# I didn't remove records according to Null values
# Since there are several fillin practices for that
# and there missig professional contex to understand which columns are must-have or not


# In[12]:


# sample code for removing all Null records from sources and add to dedicated 'BAD' records sets

# for src in src_list: 
#     cmd  = r'''{src}_BAD = {src}[{src}.isnull().any(axis=1)]'''.format(src=src)
#     exec(cmd)
#     cmd  = r'''index_with_null = {src}.index[{src}.isnull().any(axis=1)]'''.format(src=src)
#     exec(cmd)
#     cmd  = r'''{src}.drop(index_with_null,0, inplace=True)'''.format(src=src)
#     exec(cmd)


# In[13]:


def check_df_stats(df):
    cols = df.select_dtypes(include='float64').columns.tolist()
    df_dict = {}
    for c in cols:
        df_dict[c] = {'mean' : np.mean(df[c]), 'std' : np.std(df[c])}    
    
    def check_value_stats(v):
        threshold = 3
        mean = df_dict[c]['mean']
        std = df_dict[c]['std']
        z_score= (v - mean)/std
        if np.abs(z_score) > threshold: return 0
        else: return 1
        
    for c in cols:
        df[c+'_c'] = df[c].apply(check_value_stats)
        
    cols_c = [c+'_c' for c in cols]
    df['stats_check'] = df[cols_c].sum(axis=1)==len(cols_c)
    # df[df['stats_check']==True]
    
    return


# ## climate_df_Recovered_Sheet1

# In[14]:


# fixing climate_df columns header
climate_df_Recovered_Sheet1.columns = climate_df_Recovered_Sheet1.columns.tolist()[:2]+climate_df_Recovered_Sheet1.iloc[0,2:].values.tolist()

# removing climate_df_Recovered_Sheet1 invalid rows
climate_df_Recovered_Sheet1_BAD = climate_df_Recovered_Sheet1.iloc[:2,:].copy()
climate_df_Recovered_Sheet1 = climate_df_Recovered_Sheet1.iloc[2:,:]
# climate_df_Recovered_Sheet1.head(2)

# data types relevant to values, reformat and remove mismatches if needed
climate_df_Recovered_Sheet1 = climate_df_Recovered_Sheet1.astype({k:'float64' for k in climate_df_Recovered_Sheet1.columns[:6]})
climate_df_Recovered_Sheet1['HR'] = pd.to_datetime(climate_df_Recovered_Sheet1['Hr'], format='%H:%M:%S').dt.time
del climate_df_Recovered_Sheet1['Hr']
climate_df_Recovered_Sheet1['DATE'] = pd.to_datetime(climate_df_Recovered_Sheet1['Date'], format='%yyyy-%mm:%dd')
del climate_df_Recovered_Sheet1['Date']
display(climate_df_Recovered_Sheet1.head(2))
climate_df_Recovered_Sheet1.info()


# In[15]:


# check Date delta

climate_df_Recovered_Sheet1['date_diff'] = climate_df_Recovered_Sheet1['DATE'].dt.date.diff()
climate_df_Recovered_Sheet1['date_diff'].iloc[0] = pd.Timedelta('1 days 00:00:00')


# add possaible rows with longer date diff (of 1 day) to BAD_df
climate_df_Recovered_Sheet1_BAD = pd.concat([climate_df_Recovered_Sheet1_BAD
                                             , climate_df_Recovered_Sheet1[climate_df_Recovered_Sheet1['date_diff'] != pd.Timedelta('1 days 00:00:00')]])
climate_df_Recovered_Sheet1 = climate_df_Recovered_Sheet1[climate_df_Recovered_Sheet1['date_diff'] == pd.Timedelta('1 days 00:00:00')]


# In[16]:


# check_df_stats

check_df_stats(climate_df_Recovered_Sheet1)

climate_df_Recovered_Sheet1_BAD = climate_df_Recovered_Sheet1[climate_df_Recovered_Sheet1['stats_check']==False]
climate_df_Recovered_Sheet1 = climate_df_Recovered_Sheet1[climate_df_Recovered_Sheet1['stats_check']==True]
climate_df_Recovered_Sheet1.head()


# In[17]:


add_log(['low', 'success','finished work on set: climate_df_Recovered_Sheet1'])


# ***

# ### samples_df_metadata_treatments

# In[18]:


samples_df_metadata_treatments


# In[19]:


row = samples_df_metadata_treatments.iloc[[4,]].copy()
new_row = pd.concat([row,row]).reset_index(drop=True)

# redesign rows

treat_time = new_row.iloc[0,3].split('/')
treat_time = [i.strip() for i in treat_time]
for i in range(len(treat_time)):
    new_row.iloc[i,3] = treat_time[i]
    
treat_date = new_row.iloc[0,4].split('/')
treat_date = [datetime.strptime(i.replace(' ',''), "%b,%d,%Y") for i in treat_date]
for i in range(len(treat_date)):
    new_row.iloc[i,4] = treat_date[i]
    
samples_df_metadata_treatments = pd.concat([samples_df_metadata_treatments.iloc[:4,]
                                            , new_row
                                            , samples_df_metadata_treatments.iloc[5:,]]).reset_index(drop=True)
display(samples_df_metadata_treatments)


# In[20]:


row = samples_df_metadata_treatments.iloc[[10,]].copy()
new_row = pd.concat([row,row]).reset_index(drop=True)

# redesign rows

treat_time = new_row.iloc[0,3].split('/')
treat_time = [i.strip() for i in treat_time]
for i in range(len(treat_time)):
    new_row.iloc[i,3] = treat_time[i]
    
treat_date = new_row.iloc[0,4].split(',')
print(treat_date)
treat_date = [datetime.strptime(i.replace(' ',''), "%d/%m/%Y") for i in treat_date]
for i in range(len(treat_date)):
    new_row.iloc[i,4] = treat_date[i]
    
samples_df_metadata_treatments = pd.concat([samples_df_metadata_treatments.iloc[:10,]
                                            , new_row]).reset_index(drop=True)
display(samples_df_metadata_treatments)


# In[21]:


add_log(['low', 'success','finished work on set: samples_df_metadata_treatments'])


# ***

# ### samples_df_sampels

# In[22]:


# check for outlier by value counts

outlier_dict = {}
for c in samples_df_sampels.select_dtypes(include=['int64', 'object']).columns.tolist():
    df = samples_df_sampels[c].value_counts().to_frame().reset_index()
    outs = df['index'][df[c]==1].values.tolist()
    if len(outs)!=0: outlier_dict[c] = outs
        
outlier_dict.keys()


# In[23]:


outlier_dict['treatment ID']


# In[24]:


samples_df_sampels_BAD = samples_df_sampels[samples_df_sampels['treatment ID'].isin(outlier_dict['treatment ID'])]
samples_df_sampels = samples_df_sampels[~samples_df_sampels['treatment ID'].isin(outlier_dict['treatment ID'])]


# In[25]:


# check date columns in set
# looks like they have good readable format

cols = samples_df_sampels.columns.tolist()
date_cols = [c for c in cols if re.search("date", c,  re.IGNORECASE)]
samples_df_sampels[date_cols].tail(2)


# In[26]:


# object columns check

obj_cols = samples_df_sampels.select_dtypes(include=['object']).columns.tolist()
samples_df_sampels[obj_cols].head(2)


# In[27]:


obj_cols


# In[28]:


# converting

samples_df_sampels['Grain yield'] = samples_df_sampels['Grain yield'].apply(lambda x: str(x).replace(' ','.'))
samples_df_sampels['Total N content'] = samples_df_sampels['Total N content'].apply(lambda x: str(x).replace(',','.'))

samples_df_sampels = samples_df_sampels.astype({'treatment ID':'int64', 'Grain yield':'float64', 'Total N content':'float64'})


# In[29]:


# int columns check
# looks good
int_cols = samples_df_sampels.select_dtypes(include=['int64']).columns.tolist()
samples_df_sampels[int_cols].head(2)


# In[30]:


# check_df_stats

check_df_stats(samples_df_sampels)

samples_df_sampels_BAD = samples_df_sampels[samples_df_sampels['stats_check']==False]
samples_df_sampels = samples_df_sampels[samples_df_sampels['stats_check']==True]
samples_df_sampels.head()


# In[31]:


add_log(['low', 'success','finished work on set: samples_df_sampels'])


# ***

# ### sensors_df_25cm

# In[32]:


sensors_df_25cm.head()


# In[33]:


# since 'Date' stores full timestamp, then there's no need (currently) to divide to two more columns
del sensors_df_25cm['Date.1']
del sensors_df_25cm['Time']


# In[34]:


# check_df_stats

check_df_stats(sensors_df_25cm)

sensors_df_25cm_BAD = sensors_df_25cm[sensors_df_25cm['stats_check']==False]
sensors_df_25cm = sensors_df_25cm[sensors_df_25cm['stats_check']==True]
sensors_df_25cm.head()


# In[35]:


sensors_df_25cm_BAD


# In[36]:


add_log(['low', 'success','finished work on set: sensors_df_25cm'])


# ***

# # Storing to DB
# ##### SQLite is an open-source, zero-configuration, self-contained, stand-alone, transaction relational database engine

# In[37]:


add_sets = ['log_df', 'climate_df_Recovered_Sheet1_BAD', 'samples_df_sampels_BAD', 'sensors_df_25cm_BAD']
src_list += add_sets
src_list


# In[38]:


import sqlite3
conn = sqlite3.connect('noam_db.sqlite')

for s in src_list:
    cmd  = r'''{s}.to_sql('{s}', conn, if_exists="replace")'''.format(s=s)
    exec(cmd)

conn.commit()


# In[39]:


add_log(['low', 'success','finished storing all sets to DB'])

