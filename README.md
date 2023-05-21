# Цифровой прорыв 2023: VK case
## Команда "Быстрый Фурье"
[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com)

В этом репозитории представлено наше решение для задачи предсказания интенсивности взаимодействия между пользователями на платформе VK
![graph.png](misc%2Fgraph.png)

## 1. Подготовка данных.
Мы решили разбиться на несколько человек и по отдельности подготовить фичи. 
- [attr_processing.ipynb](features_generation_notebooks/attr_processing.ipynb) генерируется датасет с атрибутами и (ego_id, u, v) ключами и для каждого ego_id создается свой файл
- [graph_features.ipynb](features_generation_notebooks/graph_features.ipynb) генерируется датасет с графовыми фичами, которые не зависят от значений в связях и !!соединяет их с исходными данными
- [graph_features.ipynb](features_generation_notebooks/factorization.ipynb) генерируется датасет с фичами от факторизации

!!Внимание каждый файл генерирует свою папку с фичами ДЛЯ ОТДЕЛЬНОГО эго-графа

## 2. Модель
- [Solve_xgboost_0.247.ipynb](Solve_xgboost_0.247.ipynb) Решает задачу при помощи фичей из атрибутов и графов, чтобы получить 0.247 на паблике
- [Solve_xgboost_factorization_0.265_factorization.ipynb](Solve_xgboost_factorization_0.265_factorization.ipynb) Решает задачу при помощи фичей из атрибутов и графов и факторизации до значения 0.265 на паблике. 

Подходы которые не сработали
- [torch_mlp_train.py](torch_mlp_train.py) - Строит простую млп на графовых и атрибутных фичах. дает ~0.1 - 0.2 на паблике

Стек:
 - **Jupyter**
 - **NetworkX**: Построение фичей, анализ графов
 - **XGBoost**: Финальная модель
 - **Pytorch**: Факторизация и MLP

My contact: TG - @MalchuL