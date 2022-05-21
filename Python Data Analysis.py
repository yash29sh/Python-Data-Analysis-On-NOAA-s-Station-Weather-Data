#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math 
import collections
import urllib

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px


# In[2]:


urllib.request.urlretrieve('https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt',
                           'readme.txt')


# In[3]:


urllib.request.urlretrieve('https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt',
                           'stations.txt')


# In[4]:


stations = np.genfromtxt('stations.txt',delimiter=[11,9,10,7,3,31,4,4,6],
                                        names=['id','latitude','longitude','elevation','state','name','gsn','hcn','wmo'],
                                        dtype=['U11','d','d','d','U30','U31','U4','U4','U6'],
                                        autostrip=True)


# In[5]:


len(stations)


# In[6]:


stations


# In[7]:


pp.plot(stations['longitude'],stations['latitude'],'.',markersize=2)


# In[8]:


stations_ca = stations[stations['state']=='CA']
stations_ca


# In[9]:


pp.plot(stations_ca['longitude'],stations_ca['latitude'],'.',markersize=2)


# In[10]:


stations[stations['name']=='PASADEN']


# In[11]:


stations[np.char.find(stations['name'],'PASADENA')==0]


# In[12]:


urllib.request.urlretrieve('https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/all/USC00046719.dly', 'PASADENA.dly')


# In[13]:


# pip install getweather.getyear


# In[14]:


# pip install getyear


# In[15]:


import getweather


# In[16]:


help(getweather.getyear)


# In[17]:


getweather.getyear('PASADENA',['TMIN','TMAX'],2000)


# In[18]:


# pip install get-weather-data


# In[19]:


# pip install pyowm


# In[20]:


pasadena = getweather.getyear('PASADENA',['TMIN','TMAX'],2001)


# In[21]:


np.mean(pasadena['TMIN']),np.max(pasadena['TMIN']),np.min(pasadena['TMIN'])


# In[22]:


pasadena['TMIN']


# In[23]:


np.isnan(pasadena['TMIN'])


# In[24]:


np.nan+1


# In[25]:


False +True


# In[26]:


np.sum(np.isnan(pasadena['TMIN']))


# In[27]:


np.nanmin(pasadena['TMIN']),np.nanmax(pasadena['TMAX'])


# In[28]:


pasadena['TMIN'][np.isnan(pasadena['TMIN'])]=np.nanmean(pasadena['TMIN'])
pasadena['TMAX'][np.isnan(pasadena['TMAX'])]=np.nanmean(pasadena['TMAX'])


# In[29]:


pasadena['TMAX']


# In[30]:


pp.plot(pasadena['TMIN'])


# In[31]:


xdata=np.array([0,1,4,5,7,8],'d')
ydata=np.array([10,5,2,7,7.5,10],'d')

pp.plot(xdata,ydata,'--o')


# In[32]:


xnew=np.linspace(0,8,9)
ynew=np.interp(xnew,xdata,ydata)

pp.plot(xdata,ydata,'--o',ms=10)
pp.plot(xnew,ynew,'s')


# In[33]:


xnew=np.linspace(0,8,15)
ynew=np.interp(xnew,xdata,ydata)

pp.plot(xdata,ydata,'--o',ms=10)
pp.plot(xnew,ynew,'s')


# In[34]:


pasadena=getweather.getyear('PASADENA',['TMIN','TMAX'],2001)


# In[35]:


pasadena


# In[36]:


good =~np.isnan(pasadena['TMIN'])
x=np.arange(0,365)

np.interp(x,x[good],pasadena['TMIN'][good])


# In[37]:


def fillnans(array):
    good = ~np.isnan(array)
    x = np.arange(len(array))
    
    return np.interp(x, x[good] ,array[good])


# In[38]:


pp.plot(fillnans(pasadena['TMIN']))
pp.plot(fillnans(pasadena['TMAX']))


# In[39]:


hilo = getweather.getyear('HILO',['TMIN','TMAX'],2000)


# In[40]:


hilo['TMIN'],hilo['TMAX'] = fillnans(hilo['TMIN']),fillnans(hilo['TMAX'])


# In[41]:


np.mean(hilo['TMIN']),np.min(hilo['TMIN']),np.max(hilo['TMIN'])


# In[42]:


pp.plot(hilo['TMIN'])

for value in [np.mean(hilo['TMIN']),np.min(hilo['TMIN']),np.max(hilo['TMIN'])]:
    pp.axhline(value,linestyle='--')


# In[43]:


rain = getweather.getyear('HILO',['PRCP'],2000)['PRCP']


# In[44]:


pp.plot(rain)


# In[45]:


x = np.array([0,0,0,0,1,0,0,0,0,0,1,0,0,0])

mask = np.array([0.05,0.2,0.5,0.2,0.05])
y = np.correlate(x,mask,'same')

pp.plot(x,'o')

pp.plot(y,'x')


# In[46]:


np.ones(10)/10


# In[47]:


pp.plot(hilo['TMIN'],'.',ms=3)
pp.plot(np.correlate(hilo['TMIN'],np.ones(10)/10,'same'))


# In[48]:


pp.plot(hilo['TMIN'],'.',ms=3)
pp.plot(np.correlate(hilo['TMIN'],np.ones(10)/10,'valid'))


# In[49]:


def smooth(array,window=10,mode='valid'):
    return np.correlate(array,np.ones(window)/window,mode)


# In[50]:


pp.plot(hilo['TMIN'],'.',ms=3)
pp.plot(smooth(hilo['TMIN'],10))

pp.plot(hilo['TMAX'],'.',ms=3)
pp.plot(smooth(hilo['TMAX'],10))


# In[51]:


def plotsmooth(station,year):
    stationdata = getweather.getyear(station,['TMIN','TMAX'],year)
    
    for obs in ['TMIN','TMAX']:
        stationdata[obs] = fillnans(stationdata[obs])
        
        pp.plot(stationdata[obs])
        pp.plot(range(10,356) , smooth(stationdata[obs],20))
        
    pp.title(station)
    pp.axis(xmin=1,xmax=365,ymin=10,ymax=45)


# In[52]:


plotsmooth('HILO',2000)
plotsmooth('HILO',2001)
plotsmooth('HILO',2002)

pp.axis(ymin=15,ymax=30)


# In[53]:


pp.figure(figsize=(12,9))

for i, city in enumerate(['PASADENA','NEW YORK','SAN DIEGO','MINNEAPOLIS']):
    pp.subplot(2,2,i+1)
    plotsmooth(city,2000)


# In[54]:


pd.options.display.max_rows = 16


# In[55]:


gapminder = pd.read_csv("C:\\Users\\yashs\\Documents\\Exercise Files\\chapter6\\gapminder.csv")


# In[56]:


gapminder.head()


# In[57]:


gapminder.describe()


# In[58]:


gapminder.isna()


# In[59]:


gapminder['log_gdp_per_day']=np.log10(gapminder['gdp_per_capita']/365.25)


# In[60]:


gapminder['log_gdp_per_day']


# In[61]:


gapminder.head()


# In[62]:


import getweather


# In[63]:


allyears = np.vstack([getweather.getyear('PASADENA',['TMIN','TMAX'],year)
                     for year in range(1910,2020)])


# In[64]:


pp.matshow(allyears['TMIN'],extent=[1,365,2019,1910])
pp.colorbar()


# In[65]:


tmin_record = np.nanmin(allyears['TMIN'],axis=0)
tmax_record = np.nanmax(allyears['TMAX'],axis=0)


# In[66]:


pp.plot(tmin_record)
pp.plot(tmax_record)


# In[67]:


normal = np.vstack([getweather.getyear('PASADENA',['TMIN','TMAX'],year)
                     for year in range(1981,2011)])


# In[68]:


tmin_normal = np.nanmean(normal['TMIN'],axis=0)
tmax_normal = np.nanmean(normal['TMAX'],axis=0)


# In[69]:


pp.plot(tmin_normal)
pp.plot(tmax_normal)


# In[70]:


station,year ='PASADENA',2018
thisyear = getweather.getyear(station,['TMIN','TMAX'],year)


# In[71]:


days = np.arange(1,366)
pp.fill_between(days,thisyear['TMIN'],thisyear['TMAX'])


# In[72]:


avg = 0.5*(np.nanmean(thisyear['TMIN'])+np.nanmean(thisyear['TMAX']))


# In[73]:


avg


# In[74]:


f'{station},{year}:average temprature = {avg:.2f} C'


# In[75]:


pp.figure(figsize=(15,4.5))

pp.fill_between(days,tmin_record,tmax_record,color=(0.92,0.92,0.89),step='mid')
pp.fill_between(days,tmin_normal,tmax_normal,color=(0.78,0.72,0.72))

pp.fill_between(days,thisyear['TMIN'],thisyear['TMAX'],
               color=(0.73,0.21,0.41),alpha=0.6,step='mid')
pp.axis(xmin=1,xmax=365,ymin=15,ymax=50)

pp.title(f'{station},{year}:average temprature = {avg:.2f} C');


# In[76]:


def nyplot(station,year):
    pp.figure(figsize=(15,4.5))
    
    allyears = np.vstack([getweather.getyear('PASADENA',['TMIN','TMAX'],year)
                     for year in range(1910,2020)])
    normal = np.vstack([getweather.getyear('PASADENA',['TMIN','TMAX'],year)
                     for year in range(1981,2011)])
    tmin_record,tmax_record = np.nanmin(allyears['TMIN'],axis=0), np.nanmax(allyears['TMAX'],axis=0)
    tmin_normal,tmax_normal = np.nanmean(normal['TMIN'],axis=0), np.nanmean(normal['TMAX'],axis=0)
    
    days=np.arange(1,366)
    
    pp.fill_between(days,tmin_record,tmax_record,color=(0.92,0.92,0.89),step='mid')
    pp.fill_between(days,tmin_normal,tmax_normal,color=(0.78,0.72,0.72))
    thisyear = getweather.getyear(station,['TMIN','TMAX'],year)
    pp.fill_between(days,thisyear['TMIN'],thisyear['TMAX'],
               color=(0.73,0.21,0.41),alpha=0.6,step='mid')
    pp.axis(xmin=1,xmax=365,ymin=15,ymax=50)
    avg = 0.5*(np.nanmean(thisyear['TMIN'])+np.nanmean(thisyear['TMAX']))
    pp.title(f'{station},{year}:average temprature = {avg:.2f} C');


# In[77]:


nyplot('NEW YORK',2018)


# In[78]:


pd.options.display.max_rows = 8


# In[79]:


import zipfile


# In[80]:


zipfile.ZipFile("C:\\Users\\yashs\\Documents\\Exercise Files\\chapter7\\names.zip").extractall('.')


# In[81]:


ls


# In[82]:


ls names


# In[85]:


open("C:\\Users\\yashs\\Desktop\\names/yob2011.txt",'r').readlines()[:10]


# In[88]:


pd.read_csv("C:\\Users\\yashs\\Desktop\\names/yob2011.txt",names=['name','sex','number']).assign(year=2011)


# In[89]:


allyears=pd.concat(pd.read_csv(f"C:\\Users\\yashs\\Desktop\\names/yob{year}.txt",names=['name','sex','number']).assign(year=year)
                  for year in range(1880,2019))


# In[90]:


allyears.info()


# In[92]:


allyears.year.min(),allyears.year.max()


# In[93]:


allyears.to_csv('allyears.csv.gz',index=False)


# In[94]:


pd.options.display.max_rows = 6


# In[95]:


allyears = pd.read_csv('allyears.csv.gz')


# In[96]:


allyears


# In[181]:


allyears.groupby('year')


# In[150]:


allyears.sort_values('year',ascending=False)


# In[97]:


allyears_indexed = allyears.set_index(['sex','name','year']).sort_index()


# In[98]:


allyears_indexed.loc[('F','Mary')]


# In[99]:


pp.plot(allyears_indexed.loc[('F','Mary')])


# In[100]:


pp.plot(allyears_indexed.loc[('F','Mary')]/ allyears.groupby('year').sum())


# In[104]:


def plotname(sex,name):
    data =  allyears_indexed.loc[(sex,name)]
    
    pp.plot(data.index,data.values,label=name)
    pp.axis(xmin=1880,xmax=2018)


# In[106]:


def comparenames(sex,names):
    pp.figure(figsize=(12,2.5))
    
    for name in names:
        plotname(sex,name)
        
    pp.legend()


# In[107]:


comparenames('M',['Michael','John','David','Martin'])


# In[109]:


comparenames('F',['Emily','Anna','Claire','Elizabeth'])


# In[110]:


claries = ['Claire','Clare','Clara','Chiara','Ciara']


# In[112]:


comparenames('F',claries)


# In[116]:


allyears_indexed.loc[('F',claries),:]


# In[117]:


allyears_indexed.loc[('F',claries),:].unstack(level=2)


# In[118]:


allyears_indexed.loc[('F',claries),:].unstack(level=1)


# In[126]:


pp.figure(figsize=(12,2.5))
pp.stackplot(range(1880,2019),
            allyears_indexed.loc[('F',claries),:].unstack(level=2));


# In[124]:


pp.figure(figsize=(12,2.5))
pp.stackplot(range(1880,2019),
            allyears_indexed.loc[('F',claries),:].unstack(level=2).fillna(0),
            labels=claries);
pp.legend(loc='upper left')
pp.axis(xmin=1880,xmax=2018);

