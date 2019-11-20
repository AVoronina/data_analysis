from pandas import read_csv, DataFrame
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = read_csv('ENB2012_data.csv', ';', decimal=',')
dataset.head()

dataset = dataset.drop(dataset.columns[[10, 11]], axis='columns')
print(dataset.corr())
dataset = dataset.drop(['X1', 'X4'], axis=1)
dataset.head()

target_data = dataset['Y1']
train_data = dataset.drop(['Y1', 'Y2'], axis=1)


logistic_model = LogisticRegression()
ridge_model = Ridge(alpha=1.0)
Xtrn, Xtest, Ytrn, Ytest = train_test_split(train_data, target_data,
                                            test_size=0.4)

# Обучаем модели
logistic_model.fit(Xtrn, Ytrn.astype(int))
ridge_model.fit(Xtrn, Ytrn)
# Вычисляем коэффициент детерминации
logistic_predict = logistic_model.predict(Xtest)
ridge_predict = ridge_model.predict(Xtest)
r2_logistic = r2_score(Ytest, logistic_predict)
r2_ridge = r2_score(Ytest, ridge_predict)
