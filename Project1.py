import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

HPI = pd.read_csv('CSUSHPISA.csv')

GDP = pd.read_csv('GDP.csv')
BORROW = pd.read_csv('BORROW.csv') # Total Borrowings from the Federal Reserve
EMRATIO = pd.read_csv('EMRATIO.csv') #employment rate
HOUSEHOLD = pd.read_csv('TTLHHM156N.csv')#Household Estimates
HOUST = pd.read_csv('HOUST.csv') #New Privately-Owned Housing Units Started: Total Units
INCOME = pd.read_csv('DSPIC96.csv') #Real Disposable Personal Income
M2 = pd.read_csv('WM2NS.csv')
MORTGAGERATE = pd.read_csv('MORTGAGE30US.csv')#30-Year Fixed Rate Mortgage Average in the United States
PCE = pd.read_csv('PCE.csv') #Personal Consumption Expenditure
POP = pd.read_csv('POPTHM.csv')
TDSP = pd.read_csv('TDSP.csv') #Household Debt Service Payments as a Percent of Disposable Personal Income
UNRATE = pd.read_csv('UNRATE.csv')
WAGE = pd.read_csv('LES1252881600Q.csv')



def NewDate(x):
    return x[:4]+'-'+x[5:7]+'-'+'01'

def distribute(data, name):
    res = { }
    res[name] = [data[name].mean()]

    result = pd.DataFrame(res)
    return result

M2['DATE1'] = M2['DATE'].apply(lambda x: NewDate(x))
M2_1 = M2.groupby(['DATE1']).apply(lambda x: distribute(x,'WM2NS'))
M2_1 = M2_1.reset_index()
M2_1 = M2_1.drop('level_1', axis=1)
M2_1.columns = ['DATE', 'WM2NS']

MORTGAGERATE['DATE1'] = MORTGAGERATE['DATE'].apply(lambda x: NewDate(x))
MORTGAGERATE_1 = MORTGAGERATE.groupby(['DATE1']).apply(lambda x: distribute(x,'MORTGAGE30US'))
MORTGAGERATE_1 = MORTGAGERATE_1.reset_index()
MORTGAGERATE_1 = MORTGAGERATE_1.drop('level_1', axis=1)
MORTGAGERATE_1.columns = ['DATE', 'MORTGAGE30US']

def GetMonthlyData(data, data1, name):
    data['d'+name] = data[name].shift(-1) - data[name]
    data['d'+name] = data['d'+name].fillna(0)
    data[name+'_1'] = data[name] + data['d'+name] / 3
    data[name+'_2'] = data[name] + data['d'+name] * 2 / 3
    df = pd.merge(data1, data, how='left', on='DATE')
    df[name+'_11'] = df[name+'_1'].shift(1)
    df[name+'_21'] = df[name+'_2'].shift(2)
    loc1 = np.where(df[name].isna() & df[name+'_21'].isna())
    loc2 = np.where(df[name].isna() & df[name + '_11'].isna())
    df.loc[loc1[0], name] = df.loc[loc1[0], name+'_11']
    df.loc[loc2[0], name] = df.loc[loc2[0], name + '_21']
    result = df[['DATE', name]]
    return result


GDP_1 = GetMonthlyData(GDP, HPI, 'GDP')
WAGE_1 = GetMonthlyData(WAGE, HPI, 'LES1252881600Q')
TDSP_1 = GetMonthlyData(TDSP, HPI, 'TDSP')


data = pd.merge(HPI, GDP_1, how='left', on='DATE')
data = pd.merge(data, BORROW, how='left', on='DATE')
data = pd.merge(data, EMRATIO, how='left', on='DATE')
data = pd.merge(data, HOUSEHOLD, how='left', on='DATE')
data = pd.merge(data, HOUST, how='left', on='DATE')
data = pd.merge(data, INCOME, how='left', on='DATE')
data = pd.merge(data, M2_1, how='left', on='DATE')
data = pd.merge(data, MORTGAGERATE_1, how='left', on='DATE')
data = pd.merge(data, PCE, how='left', on='DATE')
data = pd.merge(data, POP, how='left', on='DATE')
data = pd.merge(data, TDSP_1, how='left', on='DATE')
data = pd.merge(data, UNRATE, how='left', on='DATE')
data = pd.merge(data, WAGE_1, how='left', on='DATE')
data.columns = ['DATE', 'HPI', 'GDP', 'BORROW', 'EMRATIO', 'HOUSEHOLD', 'HOUST', 'INCOME', 'M2', 'MORTGAGERATE', 'PCE', 'POP', 'TDSP', 'UNRATE', 'WAGE']
data = data.dropna()
data['HOUSEHOLD'] = data['HOUSEHOLD'].astype('float')

data = data.drop(columns=['DATE'])
mean = data.mean(axis=0)
std = data.std(axis=0)
upper = mean + 3*std
lower = mean - 3*std

loc = []
for i in range(data.shape[1]):
    loc = loc + list(np.where((data.iloc[:,i]>upper[i]) | (data.iloc[:,i]<lower[i]))[0])
loc = list(set(loc))
data = data.drop(loc)
data = data.set_index(pd.Series(range(data.shape[0])))

corr1 = np.corrcoef(data[['HPI', 'GDP', 'BORROW', 'EMRATIO', 'HOUSEHOLD', 'HOUST', 'INCOME', 'M2', 'MORTGAGERATE', 'PCE', 'POP', 'TDSP', 'UNRATE', 'WAGE']].T)

N = data.shape[0]
trainingset = data.iloc[:int(N * 0.8), :]
validationset = data.iloc[int(N*0.8):, :]
validationset = validationset.set_index(pd.Series(range(validationset.shape[0])))

mean1 = trainingset.mean(axis=0)
std1 = trainingset.std(axis=0)

for i in range(trainingset.shape[1]):
    trainingset.iloc[:, i] = (trainingset.iloc[:, i] - mean1[i]) / std1[i]
    validationset.iloc[:, i] = (validationset.iloc[:, i] - mean1[i]) / std1[i]

X_train = trainingset.iloc[:, 2:].values
y_train = trainingset.iloc[:, 1].values

X_validation = validationset.iloc[:, 2:].values
y_validation = validationset.iloc[:, 1].values

ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_train_pred = ridge.predict(X_train)
y_validation_pred = ridge.predict(X_validation)
print(ridge.coef_)
print(ridge.score(X_validation,y_validation))
mse = mean_squared_error(y_validation, y_validation_pred)
rmse = np.sqrt(mse)
print(rmse)
mae = np.sum(np.absolute(y_validation-y_validation_pred))/len(y_validation)
print(mae)


