# -*- coding: utf-8 -*-
"""

CMSE830 - Foundations of DS 
Midterm Project 

By: Yunus Shariff

"""

import pandas as pd
import numpy as np
import streamlit as st
# import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
# import hiplot as hip
import missingno as msno
#import math
import re
#from sklearn.impute import KNNImputer
# import plotly.figure_factory as ff
from operator import add
# fr om scipy import fft
from tqdm.notebook import tnrange
from tqdm import tqdm
# from statsmodels.graphics.tsaplots import plot_acf
import datetime 
import calendar

header  = st.container()
dataset = st.container()
eda = st.container()
features = st.container()
trends = st.container()

# model_training = st.container()

# st.markdown(
#     """
#     <style>
#     .main {
#      background-clor: #A2CFFE;
#      }
#     </style>
#     """
#     unsafe_allow_html = True
#     )

# @st.cache(allow_output_mutation=True)
# def get_data(filename):
#     dataset_found = pd.read_csv(filename)
#     return dataset_found
    
wiki = pd.read_csv('https://media.githubusercontent.com/media/yunus-shariff/Wiki-Forecasting/main/sample_set.csv')
org_wiki = wiki.copy()



with header:
    st.title('CMSE830 - Midterm Project')
    st.subheader('Author: Yunus Shariff')
    st.text("The project aims to use Wikipedia's traffic between July 2015 and Dec 2016,")
    st.text("to retrieve statistical information to identify trends in user traffic.")
    st.text("There is a lot that could have been accomplished with this dataset.")
    st.text("I chose to focus on monitoring daily traffic to identify patterns to help boost user")
    st.text("visits.")

#######################
# Code Block 1 begins
    
def strip_info(page, index):
    
    match = re.search(r'(\S+).wiki.edia.org_(\S+)_(\S+)',page)
    
    if match:
        if match.group(1)[-7:].lower() == 'commons':
            wiki.loc[index,'Page_Name'] = match.group(1)[:-8]
            wiki.loc[index,'Language'] = 'N/A'
        else:   
            wiki.loc[index,'Page_Name'] = match.group(1)[:-3]
            if match.group(1)[-2:].upper() in ['ZH','RU','EN','ES', 'FR','DE','JA']: 
                wiki.loc[index,'Language'] = match.group(1)[-2:].upper() 
            else: 
                wiki.loc[index,'Language'] = 'N/A'
        wiki.loc[index,'Access_Type'] = match.group(2)
        wiki.loc[index,'Agent_Type'] = match.group(3)
    
    else:
            match = re.search(r'(\S+).org_(\S+)_(\S+)',page)
            wiki.loc[index,'Page_Name'] ='N/A'
            wiki.loc[index,'Language'] = 'N/A'
            wiki.loc[index,'Access_Type'] = match.group(2)
            wiki.loc[index,'Agent_Type'] = match.group(3)

for i in range(len(wiki)):
    strip_info(wiki.loc[i,'Page'], i)

    
# Rearrange columns in desired order

cols = wiki.columns.values.tolist()
cols =  cols[0:1]+ cols[-4:] + cols[1:-4]    
wiki = wiki[cols]

# Code Block 1 ends
    
######################

# Code Block 2 begins
    

isnull_cols= wiki.isna().sum()
isnull_rows = wiki.isna().sum(axis = 1)

isnull_rows = isnull_rows.sort_values(ascending = False)
isnull_cols = isnull_cols.sort_values(ascending = False)

year = {'2015': 0,
            '2016': 0
           }

month_year= {}

months = {'01': 'January', '02': 'February', '03': 'March', '04':'April', '05':'May', '06':'June', '07':'July',
          '08':'August', '09':'September', '10':'October', '11':'November', '12':'December'}


for i in range(len(isnull_cols)):
    pattern = re.search('(\w+)-(\w+)-(\w+)',isnull_cols.index[i])
    if pattern: 
        yr = pattern.group(1)
        month = pattern.group(2)
    
        if str(months[month] + ' ' + yr) in month_year.keys():
            month_year[str(months[month] + ' ' + yr)] +=1
        else:
            month_year.update({str(months[month] + ' ' + yr) :1})

        year[yr] +=1

# Code Block 2 ends
    
####################


   
with dataset:
    st.header('Wiki Traffic Forecasting Dataset')
    st.text(f'The dataset has {str(len(wiki))[0:3]}K rows and {len(wiki.columns)} columns.')
    st.text("Let's peek into what it looks like")
    st.write(org_wiki.head(12))
    
    
with eda:
    
    
    st.subheader('EDA:')
    st.text("Here we analyze the raw data, mainly interpretting the missing values")
    st.text('Consider the matrix below. What can you infer from this figure?')
    f = msno.matrix(org_wiki.iloc[:100,:31], sort = 'ascending')
    st.pyplot(f.figure)
    
    st.text('The missingness matrix displays traffic for less than 1% of the pages during the month of')
    st.text(' July 2015')
    st.text('Traditiional forms of EDA cannot be applied to panel data.')
    st.text('Instead we extrapolate identifiers to help catalogue the desired metrics')    
    
    sel_col, disp_col = st.columns(2)
    
    plot_type = sel_col.radio('View missing values distributed on a yearly or monthly basis', ('Yearly','Monthly'), horizontal = True)
    if plot_type == 'Yearly':
        st.markdown('#### Missing values based on year')
        st.bar_chart(pd.DataFrame.from_dict(year, orient='index', columns=['Missing entries']))
        st.text('Missing entries have almost doubled in 2016')
        
    elif plot_type == 'Monthly':
        st.markdown('#### Monthly distribution of missing values')
        month_year_df = pd.DataFrame(month_year.items(), columns = ['Month-Year','Count'])
        fig = plt.figure(figsize = (9,7))
        sns.barplot(month_year_df, y = month_year_df['Month-Year'], x = month_year_df['Count'], palette = 'tab20c', dodge = False)
        st.pyplot(fig)
        st.text('Missing entries have almost doubled in 2016')
    
    #month_year_df = st.dataframe(month_year_df)
    #fig = ff.create_distplot([month_year_df[c] for c in month_year_df.columns], month_year_df.columns, bin_size=.25)
    # st.plotly_chart(fig)
    # c = alt.Chart(month_year_df).mark_circle().encode()
    # st.altair_chart(c, use_container_width=True)       
    
    st.text('The page with the most number of missing logs is:') 
    st.write(wiki.loc[isnull_rows.index[0],'Page_Name'])
    st.text('The date with the most missing values is:')
    st.write(isnull_cols.index[0])
        
    missing_val = wiki.iloc[:,1:].isnull().sum().sum()
    total_size = org_wiki.iloc[:,1:].shape[0]*org_wiki.iloc[:,1:].shape[1]

    st.text(f"Total missing values account for {missing_val/total_size:.2%} of the original dataset")
    
    sel_col1, disp_col1 = st.columns(2)
    missingness = sel_col1.selectbox('What do you think the missingness type is?', options = ['', 'MCAR', 'MAR', 'MNAR'])
    if missingness == 'MNAR':
        st.write('Aww.. Nice try!')
    elif missingness == 'MAR':
        st.write('Unfortunately, your response is incorrect')
    elif missingness == 'MCAR':
        st.write('Lucky Guess!')
    else:
        st.write('')
        
    
with features:
    st.subheader('Features derived from page info')
   
    st.text("As seen above, we are dealing with panel data, where many subjects are observed over a specified duration.")
    st.text("The pages have an interesting trend, which we can use to retrieve additional information for our narrative. \n Consider the example below:")
    pd.options.display.max_colwidth = 80
    wiki.iloc[9999:10000, :1]
    
    st.markdown('#### Observations:')
    st.markdown("- Oathbreaker_(Game_of_Thrones) = `page name`")
    st.markdown("- en = `language`")
    st.markdown("- desktop = `access type`")
    st.markdown("- all-agents = `agent type`")
    st.text('We have derived 3 additional features to define our trends')
    
    sel_col2, disp_col2 = st.columns(2)
    
    st.markdown('#### See how it works:')
    row_num = sel_col2.number_input('Would you like to try this? Please enter a number between 0 and 145063',value =13335)

###################

# Code Block begins    
    page =wiki.loc[row_num, 'Page'] 
    match2 = re.search(r'(\S+).wiki.edia.org_(\S+)_(\S+)',page)
    
    if match2:
        if match2.group(1)[-7:].lower() == 'commons':
            page_name= match2.group(1)[:-8]
            language = 'N/A'
        else:
            page_name = match2.group(1)[:-3]
        if match2.group(1)[-2:].upper() in ['ZH','RU','EN','ES', 'FR','DE','JA']: 
            language = match2.group(1)[-2:].upper() 
    
        agent = match2.group(2)
        access = match2.group(3)
        
    elif match2 is None:
        page_name = 'N/A'
        language = 'N/A'

    data_sample2 = {'Page': page,
                    'Page Name':page_name, 
                    'Language': language, 
                    'Agent Type': agent,
                    'Access Type': access}
    
    df2 = pd.DataFrame(data_sample2,index = [0])
    st.write(df2)

# Code Block ends
##################
    
    
    st.text('The language feature consists of 7 languages. Unfortunately, the language set is not \n as diverse as I had hoped')
    st.table(wiki['Language'].unique())
    
    sel_col3, disp_col3 = st.columns(2)
    selected_feature = sel_col3.selectbox('Distributions of the derived features', options = ['Language', 'Access Type', 'Agent Type'])
    
    if selected_feature == 'Language':
        fig = plt.figure(figsize = (9,7))
        st.markdown('#### Language Distribution')
        lang_dict = wiki["Language"].value_counts().to_dict()
        lang_dict_copy = lang_dict.copy()
        lang_dict['English'] = lang_dict.pop("EN")
        lang_dict['Japanese'] = lang_dict.pop("JA")
        lang_dict['German'] = lang_dict.pop("DE")
        lang_dict['N/A'] = lang_dict.pop("N/A")
        lang_dict['French'] = lang_dict.pop("FR")
        lang_dict['Chinese'] = lang_dict.pop("ZH")
        lang_dict['Russian'] = lang_dict.pop("RU")
        lang_dict['Spanish'] = lang_dict.pop("ES")
        keys = list(lang_dict.keys())
        vals = list(lang_dict.values())
        palette_color = sns.color_palette('Set3')
        # plotting data on chart
        explode = [0.1, 0, 0, 0, 0, 0, 0, 0]
        plt.pie(vals, labels=keys, colors=palette_color,explode = explode, startangle = 90, autopct='%.0f%%')
        #sns.barplot(wiki, x =keys, y = vals, hue = keys, palette = 'ocean')
        st.pyplot(fig)
    
    elif selected_feature == 'Access Type':
        fig = plt.figure(figsize = (9,7))
        st.markdown('#### Access Type Distribution')
        access_dict = wiki["Access_Type"].value_counts().to_dict()
        keys = list(access_dict.keys())
        vals = list(access_dict.values())
        sns.barplot(wiki, x =keys, y = vals, hue = keys, palette = 'bone')
        st.pyplot(fig)

    elif selected_feature == 'Agent Type':
        fig = plt.figure(figsize = (9,7))
        st.markdown('#### Agent Type  Distribution')
        agent_dict = wiki["Agent_Type"].value_counts().to_dict()
        keys = list(agent_dict.keys())
        vals = list(agent_dict.values())
        sns.barplot(wiki, x =keys, y = vals, hue = keys, palette = 'RdPu')
        st.pyplot(fig)
        
with trends:
    # lang_dict = lang_dict_copy
    # lang_list = list(lang_dict.keys())
    
    st.subheader("Here is a plot which explains the trend in traffic for different languages")
    train_data = org_wiki.copy()
    Page_name=train_data['Page']
    train_data=train_data.drop(columns=['Page']).interpolate(axis=1)
    train_data.fillna(0,inplace=True)
    train_data.insert(loc=0,column='Page',value=Page_name)
    
    lang=set()
    # st.write(lang)
    for k in Page_name:
        index=k.find('.wikipedia')
        lang.add(k[index-1:index-3:-1][::-1])
    
    lang_list=list(lang)
    # lang_traffic=[]
    lang_number = []
    data_list=[]
    for i in range(len(lang_list)):
        data_list.append(np.zeros(train_data.shape[1]-1))
    
    for i in tnrange(len(Page_name)):
        index=Page_name[i].find('.wikipedia')
        temp=lang_list.index(Page_name[i][index-1:index-3:-1][::-1])
        add_list=train_data.iloc[i].values[1:]
        data_list[temp]=list(map(add,data_list[temp],add_list))
        # lang_number[temp]=lang_number[temp] + 1
      
    lang_data=pd.DataFrame(data_list,index=lang_list,columns=train_data.columns.values[1:])
    fig = plt.figure(figsize = (9,7))
    sns.lineplot(lang_data.transpose())
    plt.xlabel('Date for which data was recorded')
    plt.ylabel('Total traffic(10^8)')
    plt.legend(labels =['Chinese','Spanish','French','Russian','English','NA','German','Japanese'])
    st.pyplot(fig)
    
    st.markdown("The Media audience contributes the most in terms of traffic even though the number of media articles is tied for fourth place")
    st.markdown("Although there are a lot more inferences to be drawn in terms of languages and a combination of access type, let's answer some burning questions!")
    
    st.markdown("#### Does this count as trivia?")
    
    month=[]
    for i in tnrange(12):
        k=[]
        month.append(k)
    for col in tqdm(train_data.columns[1:]):
        index=int(col.split('-')[1])
        month[index-1].append(np.median(train_data[col].values))
        
    for i in range(len(month)):
        month[i]=np.median(month[i])
    # st.write(month)
    if month.index(max(month))== 4:
        st.markdown("Month with most number of average visitor is: `May`") 
        st.write("Average visitors in the month of May: ", max(month) )
   
    week_day=[]
    for i in tnrange(7):
        k=[]
        week_day.append(k)
    for col in tqdm(train_data.columns[1:]):
        index=datetime.datetime.strptime(' '.join(col.split('-')), '%Y %m %d').weekday() 
        week_day[index].append(np.median(train_data[col].values))
   
    for i in range(len(week_day)):
        week_day[i]=np.median(week_day[i])
    # st.write(week_day)    
    if calendar.day_name[week_day.index(max(week_day))] == 'Sunday':
        st.markdown("Day with maximum visitors on average: `Sunday`")

    
    st.write('Average Visitors on weekdays : ',np.median(week_day[0]+week_day[1]+week_day[2]+week_day[3]+week_day[4]))
    st.write('Average Visitors on the weekend : ',np.median(week_day[5]+week_day[6]))
    
    access_data_list=[]
    for i in range(3):
      access_data_list.append([])
    for i in tnrange(len(Page_name)):
      add_list=train_data.iloc[i].values[1:]
      k=max([i.start() for i in re.finditer('org_',Page_name[i])])   
      if('all-access' in Page_name[i][k:]):
        access_data_list[0].append(np.median(add_list))
      if('desktop' in Page_name[i][k:]):
        access_data_list[1].append(np.median(add_list))
      if('mobile' in Page_name[i][k:]): 
        access_data_list[2].append(np.median(add_list))
    
    for i in range(len(access_data_list)):
        access_data_list[i]=np.median(access_data_list[i])
    st.write('Average traffic by all_access',access_data_list[0])   
    st.write('Average traffic by desktop',access_data_list[1])
    st.write('Average traffic by  mobile',access_data_list[2])
    
    month=[]
    for i in tnrange(2):
        month.append([])
    for col in tqdm(train_data.columns[1:]):
        index=int(col.split('-')[1])
        if(index<10):
            month[0].append(np.median(train_data[col].values))
        else:
            month[1].append(np.median(train_data[col].values))
    
    for i in range(len(month)):
        month[i]=np.median(month[i])
    st.write('Average number of visitors during the holidays:', month)    
    
# with model_training:
    
#     st.header('Coming Soon... to a classroom near you!')
    
      # sel_col, disp_col = st.columns(2)
#     rating = sel_col.slider('On a scale from 1 to 10, with 10 being the highest what would you rate this project? Be as brutal as you wish', min_value = -12, max_value = 10, value = 0, step = 1)
#     if rating <= 5 and rating > 0:
#         print('Tough but fair.')
#     elif rating <0:
#         print('FATALITY!')
#     elif rating > 5:
#         print("High Praise!")
#     else:
#         print('')
