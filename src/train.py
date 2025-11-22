import numpy as np
import pickle
import json
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Загрузка параметров
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Чтение обработанных данных
X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

k = params["train"]["k"]

# Обучение
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Предсказание и метрики
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# Сохранение модели
with open("model.pkl", "wb") as f:
    pickle.dump(knn, f)

# Сохранение метрик (для DVC + MLflow в будущем)
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall)
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
