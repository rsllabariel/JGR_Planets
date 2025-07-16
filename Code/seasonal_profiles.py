import numpy as np
import pandas as pd
%matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import datetime
from scipy.optimize import curve_fit
import seaborn as sns
sns.set()


# raw_data
df= pd.read_csv(r'data.csv', index_col=0) 
df.index= pd.to_datetime(df.index)

# excluding data failure':
data.loc[:'2016-06-17', '85m']= np.nan
data.loc['2016-12-14' : '2017-06-21', '85m']= np.nan

def calc_mean(df): 
    return df.groupby([df.index.hour, df.index.minute]).agg(np.mean).round(2).reset_index(drop=True) # agg(np.mean) beacause mean() gave nan

def calc_median(df):    
    return df.groupby([df.index.hour, df.index.minute]).agg(np.median).round(2).reset_index(drop=True) # agg(np.mean) beacause mean() gave nan
    
def calc_std(df):    
        return 0.5* df.groupby([df.index.hour, df.index.minute]).std().round(2).reset_index(drop=True)


# Division to Season According Temperature
df= data.between_time('08:00', '20:00')

t1= df['temp0'].resample('D').max()
t2= t1.groupby([t1.index.month, t1.index.day]).mean().reset_index(drop=True)
t2_std= t1.groupby([t1.index.month, t1.index.day]).std().reset_index(drop=True)

# Daily profiles
df_temp= data.copy()
df_temp.loc['2017-01-01':, '85m']= np.nan

df= df_temp.copy()

summer = df[df.index.month.isin([5, 6,7,8, 9, 10])]
winter = df[df.index.month.isin([11, 12,1,2, 3,4])]
day_summer, day_winter = [calc_mean(df) for df in [summer, winter]]
std_summer, std_winter = [calc_std(df) for df in [summer, winter]] 
