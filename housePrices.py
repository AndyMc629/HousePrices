import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix

#Function to normalise data columns to range (0,1).
def normalise(df):
	result = df.copy()
	for feature_name in df.columns:
        	max_value = df[feature_name].astype(np.float32).max()
        	min_value = df[feature_name].astype(np.float32).min()
        	result[feature_name] = (df[feature_name].astype(np.float32) - min_value) / (max_value - min_value)
    	return result	

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CLEAN UP DATA
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#read in data. Column names from http://bit.ly/1hh97JI
data=pd.read_csv("pp-2015-part1.csv", names=["TUI","Price","DateOfTransfer","Postcode","PropertyType","Old/New", "Duration","PAON", "SAON","Street","Locality","Town","District","County","PPDCategoryType","RecordStatus"])

#drop unwanted/un-needed data. Axis=1 -> column.
data.drop(['TUI','Postcode','Old/New','PAON','SAON', 'Street','Locality', 'District','County','PPDCategoryType','RecordStatus'],axis=1,inplace=True) 

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PREPARE DATA FOR ANALYSIS
#
# Mappings: 
# in London = 1, out of London = 0
# lease duration => L=0, F=1
# PropertyVals={'S':'1','T':'2','F':'3', 'D':'4', 'O':'5'}
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#convert dates to number of days since start of data.
data['DateOfTransfer'] = pd.to_datetime(data['DateOfTransfer'])
data['DateOfTransfer']=(data['DateOfTransfer'] - data['DateOfTransfer'].min())/np.timedelta64(1,'D')

#map to in and out of London values
data['InLondon'] = ['1' if x=="LONDON" else '0' for x in data.Town]
data.drop('Town',axis=1,inplace=True)

#map duration values
data['Duration']=['1' if x=='F' else '0' for x in data.Duration]

#define a dictionary of the values and call map.
#changes property types to PropertyVals in place.
#Need this when not binary options like above.
PropertyVals={'S':'1','T':'2','F':'3', 'D':'4', 'O':'5'}
data['PropertyType']=data['PropertyType'].map(PropertyVals)

#remove rows where house prices > 3 std_dev.
data = data.drop(data[data.Price>(data.Price.mean()+3*data.Price.std())].index)

#quick check for nans ....
nanIndex=pd.isnull(data).any(1).nonzero()[0]
print "nans on rows \n", data.ix[nanIndex]

##Plot price histogram with data coloured by in London or outside London.
#plt.figure()
#plt.xlim(0,2000)#remember price is in k pounds now.
#data[data.InLondon=='1'].Price.hist(bins=1000,alpha=0.4,color='red',label='London')
#data[data.InLondon=='0'].Price.hist(bins=1000,alpha=0.4,color='blue',label='rUK')
#plt.xlabel('Price (k)')
#plt.legend()
#plt.savefig("PriceHisto.pdf")

##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## DATA CLEANED AND SUMMARY STATS PLOTTED
## APPLY ML ALGO
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## NORMALISE, SPLIT INTO TEST/TRAIN
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##Normalise
#data=(data-data.min().astype(np.float32))/(data.max()-data.min())
data=normalise(data)

##Some more figures .. 
#plt.figure()
#data.loc[:,['Price','DateOfTransfer']].plot(kind='box')
#plt.savefig("Boxplot.pdf")
#plt.close()

#plt.figure()
#data.hist()
#plt.savefig("AllHists.pdf")
#plt.close()

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


##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## MODEL - NOTE: I PLAYED AROUND WITH A FEW FOR FUN.
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##Uncomment below if you want kfold cross validation.
#from sklearn.cross_validation import *

##function to train the data and output Coefficient of det on train.
def train_and_evaluate(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    print "Coefficient of determination on training set:",clf.score(X_train, y_train)
    Coeffs=linear.coef_
    print "Coeffs=", Coeffs
    #xtics = dataframe column names
    labels=list(train.columns.values)
    print "labels",labels[1:]
    x=np.linspace(1,Coeffs.size,Coeffs.size)
    plt.figure()
    plt.xlim(x[0]-0.5,x[-1]+0.5)
    plt.plot(x,Coeffs, linestyle='None',marker='D',markersize=10)
    plt.xticks(x,labels[1:])
    plt.savefig("Coeffs.pdf")
    plt.close()
    # create a k-fold cross validation iterator of k=5 folds
    #cv = KFold(X_train.shape[0], 5, shuffle=True,random_state=33)
    #scores = cross_val_score(clf, X_train, y_train, cv=cv)
    #print "Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores)

##LINEAR REGRESSION
from sklearn import linear_model
#create lin reg object.
linear=linear_model.LinearRegression()
##ridge_regression
#ridge=linear_model.Ridge(alpha=1.0)
##lasso_regression
#lasso=linear_model.Lasso(alpha=1.0)
##Stochastic grad descrent regressor rather than closed form.
#linear=linear_model.SGDRegressor(loss='squared_loss', penalty=None, random_state=42)

##SVM for REGRESSION
#from sklearn import svm
#svr = svm.SVR(kernel='linear')

##RANDOM FORESTS for REGRESSION
#from sklearn import ensemble
#extraTrees=ensemble.ExtraTreesRegressor(n_estimators=10, random_state=42)

##GAUSSIAN PROCESSES
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF, ConstantKernel as C
#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#gpr=GaussianProcessRegressor(kernel=kernel,alpha=0,optimizer=None, normalize_y=True)
#gpr.fit(x_np_train,y_np_train)

##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## TRAINING AND CROSS-VALIDATION
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print "LinReg: ", train_and_evaluate(linear,x_np_train,y_np_train), "\n"
#print "ridge: ", train_and_evaluate(ridge,x_np_train,y_np_train), "\n"
#print "lasso: ", train_and_evaluate(lasso,x_np_train,y_np_train), "\n"
#print "SVR: ", train_and_evaluate(svr,x_np_train,y_np_train)
#print "Extra trees: ", train_and_evaluate(extraTrees, x_np_train,y_np_train)
#print "gpr: ", train_and_evaluate(gpr,x_np_train, y_np_train)

##Predict Output
#predicted=linear.predict(x_np_test)

## Plot predicted vs true data.
#plt.figure()
#plt.scatter(y_np_test,predicted)
#plt.plot([y_np_test.min(), y_np_test.max()], [y_np_test.min(), y_np_test.max()], 'k--', lw=4)
#plt.xlabel('Measured')
#plt.ylabel('Predicted')
#plt.savefig("Prediction.pdf")
#plt.close()

##Rerun with most strongly correlated features 
## - PropertyType and InLondon
##convert train dataframe to numpy array
#x_np_train=train.values[:,[2,3,4]].astype(np.float32)

##convert test dataframe to numpy array
#x_np_test=test.values[:,[2,3,4]].astype(np.float32)

#print "LinReg: ", train_and_evaluate(linear,x_np_train,y_np_train), "\n"

##Predict Output
#predicted=linear.predict(x_np_test)

## Plot predicted vs true data.
#plt.figure()
#plt.scatter(y_np_test,predicted)
#plt.plot([y_np_test.min(), y_np_test.max()], [y_np_test.min(), y_np_test.max()], 'k--', lw=4)
#plt.xlabel('Measured')
#plt.ylabel('Predicted')
#plt.savefig("Prediction.pdf")
#plt.show()
#plt.close()



