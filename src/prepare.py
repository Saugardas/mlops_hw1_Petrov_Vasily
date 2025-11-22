import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загружаем параметры
with open("params.yaml") as f:
    params = yaml.safe_load(f)

iris_data = pd.read_csv("./data/raw/data.csv", sep=',')
iris_values = iris_data[['sepal.length', 'sepal.width']]
iris_target = iris_data['variety']

# Дальше подготавливаем данные для эксперимента
iris_target_encoded, class_names = pd.factorize(iris_target) # Кодирование текстовых меток целевой переменной
scaler = StandardScaler()
iris_values_scaled = scaler.fit_transform(iris_values)
X_train, X_test, y_train, y_test = train_test_split(
    iris_values_scaled,
    iris_target_encoded,
    test_size=params["prepare"]["test_size"],
    random_state=params["prepare"]["random_state"]
)

# Создаём папку, если её нет
os.makedirs("data/processed", exist_ok=True)

# Сохраняем
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)
