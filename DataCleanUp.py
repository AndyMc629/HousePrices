import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix

#read in data. Note "RecordStatus" may not be needed in full file ....
data=pd.read_csv("pp-2015-part1.csv", names=["TUI","Price","DateOfTransfer","Postcode","PropertyType","Old/New", "Duration","PAON", "SAON","Street","Locality","Town","District","County","PPDCategoryType","RecordStatus"])

#drop unwanted/un-needed data. Axis=1 -> column.
data.drop(['TUI','DateOfTransfer','Postcode','Old/New','PAON','SAON', 'Street','Locality', 'District','County','PPDCategoryType','RecordStatus'],axis=1,inplace=True) 

#
# PREPARE DATA FOR ANALYSIS
#
# in London = 1, out of London = 0
# lease duration => 0= L, 1=F
# prop type => 000=D,100=S,010=T,001=F

#map to in and out of London values
data['InLondon'] = ['1' if x=="LONDON" else '0' for x in data.Town]
data.drop('Town',axis=1,inplace=True)

#map duration values
data['Duration']=['1' if x=='F' else '0' for x in data.Duration]

"""
#plot old data before changing (easier to read)
plt.figure()
data.PropertyType.value_counts().plot(kind='bar')
plt.savefig("PropTypeHisto.pdf")
"""
#define a dictionary of the values and call map.
#changes property types to PropertyVals in place.
#Need this when not binary options like above.
PropertyVals={'S':'100','T':'101','F':'001', 'D':'000'}
data['PropertyType']=data['PropertyType'].map(PropertyVals)

data.Price=data.Price/1000
data.Price=data.Price.round()


#quick check.
print data

"""
plt.figure()
scatter_matrix(data,alpha=0.2,figsize=(6,6), diagonal='kde')
plt.savefig("ScatterMatrix.pdf")
"""


plt.figure(1)
plt.xlim(0,2000)#remember price is in k pounds now.
data[data.InLondon=='1'].Price.hist(bins=1000,alpha=0.4,color='red',label='London')
data[data.InLondon=='0'].Price.hist(bins=1000,alpha=0.4,color='blue',label='rUK')
plt.xlabel('Price (k)')
plt.legend()
plt.savefig("PriceHisto.pdf")


plt.figure(2)
plt.xlim(0,2000)#remember price is in k pounds now.
data[data.PropertyType=='100'].Price.hist(bins=1000,alpha=0.4,color='k',label='Semi-detached')
data[data.PropertyType=='101'].Price.hist(bins=1000,alpha=0.4,color='red',label='Terraced')
data[data.PropertyType=='001'].Price.hist(bins=1000,alpha=0.4,color='blue',label='Flat')
data[data.PropertyType=='000'].Price.hist(bins=1000,alpha=0.4,color='green',label='Detached')
plt.xlabel('Price (k)')
plt.legend()
plt.savefig("TypeHisto.pdf")

plt.figure(3)
data.boxplot(column='Price',by='InLondon')
plt.savefig("Boxplot.pdf")

