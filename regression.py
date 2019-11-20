import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# Чтение файла с данными
dataset = pd.read_csv('ENB2012_data.csv', ';', decimal=',')
dataset.head()
# Так как преобразовать нормально файл xlsx в csv не получилось,
# то удалим столбцы с NaN
dataset = dataset.dropna(axis=1)
# Определим корреляцию между параметрами
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
print(dataset.corr())
# Удалим из выборки параметры с корреляцией > 95 %
dataset = dataset.drop(['X1', 'X4'], axis=1)

# Зададим данные для обучения и предсказания
target_data = dataset['Y1']
train_data = dataset.drop(['Y1', 'Y2'], axis=1)

logistic_model = LogisticRegression()
ridge_model = Ridge(alpha=1.0)
X_train, X_test, Y_train, Y_test = train_test_split(
    train_data, target_data, test_size=0.4)

# Обучаем модели
logistic_model.fit(X_train, Y_train.astype(int))
ridge_model.fit(X_train, Y_train)

# Предсказываем значения
logistic_predict = logistic_model.predict(X_test)
ridge_predict = ridge_model.predict(X_test)

# Вычисляем коэффициент детерминации
r2_logistic = r2_score(Y_test, logistic_predict)
r2_ridge = r2_score(Y_test, ridge_predict)
print(r2_logistic)
print(r2_ridge)

# Строим графики спрогнозированных значений
fig, axes = plt.subplots(2, 1)
axes[0].set_title('Гребневая регрессия')
axes[1].set_title('Логистическая регрессия')
predicts = (logistic_predict, ridge_predict)
for i, predict in enumerate(predicts):
    for x in X_test:
        axes[i].scatter(X_test[x].values, predict, s=5, label=x)
    axes[i].legend()
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('Y1')

plt.show()
