import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

# --- КРОК 1: ЗАВАНТАЖЕННЯ ДАНИХ ---
print("Завантаження EMNIST (лише літери)...")
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split=['train[:10%]', 'test[:10%]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Мапа символів (цифри 0-9 йдуть першими, потім літери)
full_label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Функція фільтрації цифр та препроцесингу
def filter_and_preprocess(image, label):
    # Залишаємо лише класи >= 10 (літери)
    # Зміщуємо мітку на -10, щоб класи були від 0 до 51
    new_label = label - 10

    # Виправляємо орієнтацію зображення
    image = tf.transpose(tf.reshape(image, [28, 28]))
    image = tf.expand_dims(image, -1)
    image = tf.cast(image, tf.float32) / 255.0
    return image, new_label

def is_letter(image, label):
    return label >= 10

# --- КРОК 2: ПІДГОТОВКА ТА ФІЛЬТРАЦІЯ ---
# Фільтруємо цифри через .filter()
train_letters = ds_train.filter(is_letter).map(filter_and_preprocess)
train_data = train_letters.shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)

test_letters = ds_test.filter(is_letter).map(filter_and_preprocess)
test_data = test_letters.batch(128)

num_classes = 52
print(f"Кількість класів після фільтрації: {num_classes} (A-Z, a-z)")

# --- КРОК 3: ПОБУДОВА НЕЙРОМЕРЕЖІ ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- КРОК 4: НАВЧАННЯ ---
print("\nНавчання нейромережі...")
history = model.fit(train_data, epochs=5, validation_data=test_data.take(20))

# --- КРОК 5: ПЕРЕВІРКА ---
letters_only_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

for img, lbl in test_data.take(1):
    preds = model.predict(img)
    idx = np.random.randint(0, len(img))

    predicted_char = letters_only_map[np.argmax(preds[idx])]
    actual_char = letters_only_map[lbl[idx].numpy()]

    plt.imshow(img[idx].numpy().squeeze(), cmap='gray')
    plt.title(f"Предікт: {predicted_char} | Реально: {actual_char}")
    plt.axis('off')
    plt.show()