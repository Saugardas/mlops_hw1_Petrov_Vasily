import numpy as np
import mlflow
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
# Подключение MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Эксперимент")

with mlflow.start_run(run_name=f"knn_for_k_{k}"):
    # Логируем параметры
    mlflow.log_param("k", k)

    # Обучение
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Предсказание и метрики
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Логируем метрики
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Логируем модель
    mlflow.log_param("model", "KNeighborsClassifier")
    mlflow.sklearn.log_model(knn, name=f"model_k{k}")
