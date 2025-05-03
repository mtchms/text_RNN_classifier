import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

CSV_PATH = 'styles.csv'
IMAGES_DIR = 'images'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_FILE = 'fashion_model.keras'

def load_data():
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
    df = df.dropna(subset=['id', 'articleType'])
    df['label'] = df['articleType'].astype(str)

    counts = df['label'].value_counts()
    df = df[df['label'].isin(counts[counts >= 2].index)]

    label_mapping = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    df['label_idx'] = df['label'].map(label_mapping)

    df['full_path'] = df['id'].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

    return train_df, val_df, label_mapping

def generator(df, label_mapping, batch_size=BATCH_SIZE):
    while True:
        df = df.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            images = []
            labels = []
            for _, row in batch.iterrows():
                try:
                    img = tf.keras.utils.load_img(row['full_path'], target_size=IMG_SIZE)
                    img = tf.keras.utils.img_to_array(img)
                    img = preprocess_input(img)
                    images.append(img)
                    labels.append(row['label_idx'])
                except Exception as e:
                    print(f"Ошибка при загрузке {row['full_path']}: {e}")
            if images and labels:
                images = np.array(images)
                labels = to_categorical(labels, num_classes=len(label_mapping))
                yield images, labels

def create_model(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    train_df, val_df, label_mapping = load_data()

    train_gen = generator(train_df, label_mapping)
    val_gen = generator(val_df, label_mapping)

    steps_per_epoch = len(train_df) // BATCH_SIZE
    validation_steps = len(val_df) // BATCH_SIZE

    input_shape = IMG_SIZE + (3,)
    num_classes = len(label_mapping)

    model = create_model(input_shape, num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        ModelCheckpoint(MODEL_FILE, monitor='val_accuracy', save_best_only=True)
    ]

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2
    )

    print(f"Модель обучена и сохранена как {MODEL_FILE}")

if __name__ == '__main__':
    main()
