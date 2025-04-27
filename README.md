![image](https://github.com/user-attachments/assets/0caedbfc-76eb-4a18-ad88-e3912aba76dc)Модель классификации одежды на основе MobileNetV2
Описание проекта
Эта модель обучалась на датасете Fashion Product Images с Kaggle:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

Цель модели — по изображению одежды предсказать её категорию (articleType).
Модель принимает входы размером (128, 128, 3). На выходе — вектор вероятностей для всех классов (softmax).

Данные
Источник: Fashion Product Images Dataset (Kaggle)

Используются только те примеры, где заполнены поля id и articleType. Отбираются только те категории (articleType), в которых есть минимум 2 изображения.

Базовой сетью послужила MobileNetV2, верхние слои были удалены.

Дополнительные слои:

GlobalAveragePooling2D

Dense(128, activation='relu')

Dense(num_classes, activation='softmax')
![Без имени_1](https://github.com/user-attachments/assets/b04dab82-bb7a-4be1-9b3b-53861eaa7de1)
![Без имени](https://github.com/user-attachments/assets/0aa2b35f-9e57-45f7-8f06-f7fbf487f2d3)

Параметры обучения
Размер изображений: 128×128 пикселей

Размер батча: 32

Оптимизатор: Adam (начальная скорость обучения 0.001)

Функция потерь: categorical_crossentropy

Метрика: accuracy

Эпохи: до 20, с остановкой при отсутствии улучшений (EarlyStopping) (остановка произошла на 17 эпохе)

Коллбэки:
EarlyStopping по val_accuracy (patience=5)

ReduceLROnPlateau по val_loss (уменьшение lr в 2 раза, patience=3)

ModelCheckpoint — сохранение лучшей модели в файл fashion_model.keras


Особенности
Применяется балансировка классов через стратифицированное разбиение (train_test_split с stratify).

Используется динамическая генерация батчей через генераторы (для экономии памяти).

Лучший accuracy показанный моделью на валидационной выборке был равен 0.8752.
Ниже некоторые примеры работы модели:

![Без имени_3](https://github.com/user-attachments/assets/0486ad0b-0e06-4d2e-bdab-99ed49888111)
![Без имени_2](https://github.com/user-attachments/assets/c9de81b0-00ac-442c-aaae-1e16c9e0fad3)
