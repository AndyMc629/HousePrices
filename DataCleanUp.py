import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix

def normalise(df):
	result = df.copy()
	for feature_name in df.columns:
        	max_value = df[feature_name].astype(np.float32).max()
        	min_value = df[feature_name].astype(np.float32).min()
        	result[feature_name] = (df[feature_name].astype(np.float32) - min_value) / (max_value - min_value)
    	return result	

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CLEAN UP DATA - THEN APPLY ML ALGO
#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
#PropertyVals={'S':'100','T':'101','F':'001', 'D':'000', 'O':'010'}
PropertyVals={'S':'1','T':'2','F':'3', 'D':'4', 'O':'5'}
data['PropertyType']=data['PropertyType'].map(PropertyVals)

#Normalise everything
#data.Price=data.Price/1000
#data.Price=data.Price.round()

#remove rows where house prices > 2 mil.
#data = data.drop(data[data.Price>2000].index)

#quick check.
#print "number of nans=", data.isnull().sum().sum()
nanIndex=pd.isnull(data).any(1).nonzero()[0]
print "nans on rows \n", data.ix[nanIndex]
#print data

"""
plt.figure()
scatter_matrix(data,alpha=0.2,figsize=(6,6), diagonal='kde')
plt.savefig("ScatterMatrix.pdf")
"""

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
"""

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# DATA CLEANED AND SUMMARY STATS PLOTTED
# APPLY ML ALGO
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NORMALISE, SPLIT INTO TEST/TRAIN
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Normalise
#data=(data-data.min().astype(np.float32))/(data.max()-data.min())
data=normalise(data)

#plt.figure(3)
#data.boxplot(column='Price',by='InLondon')
#plt.savefig("Boxplot.pdf")
#plt.show()

#test/train split in Pandas.
train=data.sample(frac=0.8,random_state=200)
test=data.drop(train.index)

#convert train dataframe to numpy array
x_np_train=train.values[:,1:].astype(np.float32)
y_np_train=train.values[:,0].astype(np.float32)

#convert test dataframe to numpy array
x_np_test=test.values[:,1:].astype(np.float32)
y_np_test=test.values[:,0].astype(np.float32)
print y_np_test
print "\n \n", y_np_train


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# MODEL
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from sklearn.cross_validation import *
def train_and_evaluate(clf, X_train, y_train):
	clf.fit(X_train, y_train)
	print "Coefficient of determination on training set:",clf.score(X_train, y_train)
	# create a k-fold cross validation iterator of k=5 folds
	cv = KFold(X_train.shape[0], 10, shuffle=True,random_state=33)
	scores = cross_val_score(clf, X_train, y_train, cv=cv)
	print "Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores)

#LINEAR REGRESSION
from sklearn import linear_model
#create lin reg object.
#linear=linear_model.LinearRegression()
linear=linear_model.SGDRegressor(loss='squared_loss',
penalty=None, random_state=42)

#SVM for REGRESSION
from sklearn import svm
svr = svm.SVR(kernel='linear')

#RANDOM FORESTS for REGRESSION
from sklearn import ensemble
extraTrees=ensemble.ExtraTreesRegressor(n_estimators=10, random_state=42)

#GAUSSIAN PROCESSES
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
gpr=GaussianProcessRegressor(alpha=0.0)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TRAINING AND CROSS-VALIDATION
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#print "SGDReg: ", train_and_evaluate(linear,x_np_train,y_np_train), "\n"
#print "SVR: ", train_and_evaluate(svr,x_np_train,y_np_train)
#print "Extra trees: ", train_and_evaluate(extraTrees, x_np_train,y_np_train)
print "gpr: ", train_and_evaluate(gpr,x_np_train, y_np_train)

#Predict Output
#predicted= linear.predict(x_np_test)

#fig,ax=plt.subplots()
#ax.scatter(y_np_test,predicted)
#ax.plot([y_np_test.min(), y_np_test.max()], [y_np_test.min(), y_np_test.max()], 'k--', lw=4)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')

#fig,ax=plt.subplots()
#ax.scatter(x_np_train[:,2],y_np_train)
#ax.set_xlabel('InLondon')
#ax.set_ylabel('Price')

#plt.show()
#print predicted





