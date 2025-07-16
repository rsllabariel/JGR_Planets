import numpy as np
import pandas as pd
%matplotlib notebook
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import curve_fit
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
import plot_settings
from plot_settings import darkgray


df_org= pd.read_csv(r'data_filtered.csv', index_col=0)
df_org.index= pd.to_datetime(df_org.index)

data= df_org.copy()

# division Oct-Sept

data['year']=''
data.loc[:'2015-09-30', 'year'] ='y1'
data.loc['2015-10-01':'2016-09-30', 'year'] ='y2'
data.loc['2016-10-01':'2017-09-30', 'year'] ='y3'
data.loc['2017-10-01':'2018-09-30', 'year'] ='y4'

# regression

def calc_r2(actual, predicted, y_mean):
    ssr= np.sum( (actual- predicted)**2 )
    sst= np.sum( (actual- y_mean)**2 )
    r_squared= 1- (ssr/sst)
    return r_squared


def exp_func(x, a, b, c):
    return a* np.exp(b *x) + c

def calc_exp_regression(df, col1, col2, arr_to_fit, p0): # , intercept_max_bound
    
    df= df[[col1, col2]].dropna(how='any')
    X= df[col1].values
    Y= df[col2].values

    popt, pcov = curve_fit(exp_func, X, Y, p0=p0,  bounds=((-np.inf, -np.inf, 0), (np.inf, np.inf, Y.min()+1 )) , maxfev=1000)  # bounds so intercept>0
    a, b, c=  popt[0], popt[1], popt[2] #     a, b, c= [np.round(i, 3) for i in [a, b, c]] # can't do round otherwise a will become 0
    err = np.round(np.sqrt(np.diag(pcov)), 1) 

    line=  a *np.exp( b* arr_to_fit ) +c
    line= np.round(line, 3)
    
    line2= a * np.exp(b* X ) +c 
    r2=  calc_r2(Y, line2, Y.mean()) #   (actual, predicted, y_mean):
    
    res={'col': col2, 'a': a, 'b':b, 'c':c, 'r2': r2, 'err': err, 'var': arr_to_fit, 'line':line }
    
    return res

df_day= data.between_time('08:00', '20:00')
df_day.loc['2017-09-04':'2018-03-14']=np.nan
df_day['temp_diff']= df_day['temp0']- df_day['temp0'].min()

df= df_day[['temp0', 'temp_diff','60m']].copy()

df_max= df.resample('D').max()
df_max['year']= data['year'].resample('D').first()

summer_months= [5,6,7,8,9,10]
winter_months=[11,12,1,2,3,4]

df= df_max.copy()
summer= df[df.index.month.isin(summer_months)]
winter= df[df.index.month.isin(winter_months)]

years= ['y1', 'y2', 'y3', 'y4']

# summer dict:
y15, y16, y17, y18 = [summer[summer['year']==y ] for y in years]
summer_dict= {y: df for y, df in zip(years, [y15, y16, y17, y18])}

# winter dict: 
y15, y16, y17, y18 = [winter[winter['year']==y ] for y in years]
winter_dict= {y: df for y, df in zip(years, [y15, y16, y17, y18])}

# all
all_dict= {y: df[df['year']==y] for y in years}

df= all_dict.copy()
# col1= 'temp_diff'
col1= 'temp0'
col2='60m'


# regression line of each year
arr= np.linspace(df_max[col1].min()-0.1, df_max[col1].max()+0.1 , 100)
temp=[]
for y in years:
    
    temp2=[]
    if df[y][col2].dropna().empty:
        a= np.nan
        temp2.append(a)
        continue

    res = calc_exp_regression(df[y], col1 , col2, arr, p0=[1e-10, 0.1, 0])
    temp.append({y:res})   

year_reg= temp.copy()
year_reg60max= year_reg.copy()


# plot year
sns.set_style("ticks")
position= []

fig, ax = plt.subplots(figsize=(11, 7),  nrows= 2, ncols= 2, sharex= 'col') 
fig.subplots_adjust(bottom=0.1,
                    wspace= 0.3,
                    hspace= 0.25, 
                    top=0.85)

fig.suptitle('Relationship between Atmospheric Temperature Gradient\nand Rn at averaged 60m Depth')

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

col1='temp_diff'
col='60m'
c= 'b'

ax= ax.flatten()
for (j, y) in (enumerate(years)):

    S= summer_dict[y][col]
    W= winter_dict[y][col]
    d= year_reg[j][y]

    if S.dropna().empty & W.dropna().empty: # if i do not do continue matplotlib won't plot
        continue

    ax[j].plot(summer_dict[y][col1], S , 'o', color=c, label= 'S')
    ax[j].plot(winter_dict[y][col1], W , 'o', color=c, fillstyle='none', label='W')
    ax[j].plot(d['var'] , d['line'], c='r', linewidth=3)
    ax[j].text(0.05, 0.8,'y = {} * exp({} * \u0394T)\n$r^2$= {}'.format(d['a'], d['b'], d['r2']), color='k',transform=ax[j].transAxes,fontsize= 12)
    ax[j].set_title(y, weight='bold')
    ax[j].set_ylabel('Rn\n[kbq\m$^3$]', rotation= 0, labelpad=30,  weight='bold', fontsize=10 )
    ax[j].yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax[j].tick_params(axis='both', which='major', labelsize=10)    
    ax[0].legend(loc='center left')
   
plt.rc('ytick', labelsize=10)
plt.rc('xtick', labelsize=10)    
fig.text(0.5, 0.01, '\u0394T Temperature Gradient [K]', ha='center', rotation=0, fontsize=12, weight='bold')


# comparison all years

# regression of all data together
df= df_max.loc[:'2018-09-30'] # excluding points that are not in regression
col1= 'temp0'
arr= np.linspace(df[col1].min()-1, df[col1].max()+1, 100)

res = calc_exp_regression(df_max, col1 , '60m', arr, p0=[1e-10, 0.1, 0])
all_reg= res.copy()
years= ['y1', 'y2', 'y3', 'y4']

temp=[]
for (j, y)in (enumerate(years)):
    s= year_reg60max[j][y]['line']
    temp.append(pd.Series(s))

df_reg= pd.concat(temp, axis=1)   
df_reg['std']= df_reg.std(axis=1)
df_reg['mean']= df_reg.iloc[:,0:3].mean(axis=1)


years= ['y1', 'y2', 'y3', 'y4']
ylabels= ['Jan- Sept 2015', 'Oct 2015 - Sept 2016', 'Oct 2016 - Sept 2017', 'Mar-Sept 2018']

sns.set_style("ticks")

fig, ax = plt.subplots(figsize= (6, 4)) 
fig.subplots_adjust(bottom=0.2,
                    top=0.9,
                   left= 0.2
                   )
plt.suptitle('Regression Lines')

df= year_reg60max.copy()
d= all_reg.copy()
for (j, y), label in zip(enumerate(years), ylabels):
    
    ax.plot(df[j][y]['var'], df[j][y]['line'], linewidth=2 , label=label)

plt.plot(all_reg['var'], all_reg['line'], c='k', linewidth=3 , label= 'All Data')
plt.fill_between(all_reg['var'], (df_reg['mean'] - df_reg['std']), (df_reg['mean']+ df_reg['std']), 
                 color='b', alpha=0.3)

plt.text(0.05, 0.85,'y = {:.2E} * exp({:.2f} * T)\n$r^2$= {:.3f}'.format(d['a'], d['b'], d['r2']), 
               color='k',transform=ax.transAxes,fontsize= 12)

plt.legend(loc=(0.05, 0.4)) #(loc= 'center left')
plt.xlabel( 'T [K]', rotation=0, weight='bold') #fontsize=10, 
plt.ylabel('Rn\n[kbq\m$^3$]', rotation= 0, labelpad=30,  weight='bold' )
