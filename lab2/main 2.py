import os
import random
import warnings
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from gtts import gTTS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Вимикаємо зайві попередження для чистоти консолі
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Словник: Авіаційний алфавіт та цифри
AVIATION_ALPHABET = {
    'alpha': 'A', 'bravo': 'B', 'charlie': 'C', 'delta': 'D', 'echo': 'E',
    'foxtrot': 'F', 'golf': 'G', 'hotel': 'H', 'india': 'I', 'juliett': 'J',
    'kilo': 'K', 'lima': 'L', 'mike': 'M', 'november': 'N', 'oscar': 'O',
    'papa': 'P', 'quebec': 'Q', 'romeo': 'R', 'sierra': 'S', 'tango': 'T',
    'uniform': 'U', 'victor': 'V', 'whiskey': 'W', 'xray': 'X', 'yankee': 'Y', 'zulu': 'Z',
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
}
CLASSES = list(AVIATION_ALPHABET.keys())
DATASET_DIR = "speech_dataset"

# 2. Генерація синтетичного датасету за допомогою ШІ (gTTS) + Аугментація
def generate_dataset(samples_per_word=5):
    """Створює звукові файли для кожного слова і додає варіації (шум/зміна швидкості)."""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print("--- Генерація базових аудіофайлів через gTTS ---")

        for word in CLASSES:
            word_dir = os.path.join(DATASET_DIR, word)
            os.makedirs(word_dir, exist_ok=True)

            # Генерація оригінального голосу
            tts = gTTS(text=word, lang='en', slow=False)
            base_path = os.path.join(word_dir, f"{word}_base.wav")
            tts.save(base_path)

            # Завантажуємо базовий файл для аугментації
            y, sr = librosa.load(base_path, sr=16000)

            for i in range(samples_per_word):
                # Додаємо випадковий білий шум
                noise = np.random.randn(len(y))
                y_noisy = y + 0.005 * noise

                # Випадково змінюємо швидкість від 0.8 до 1.2
                rate = np.random.uniform(0.8, 1.2)
                y_stretched = librosa.effects.time_stretch(y_noisy, rate=rate)

                sf.write(os.path.join(word_dir, f"{word}_{i}.wav"), y_stretched, sr)
        print("Датасет успішно згенеровано та аугментовано!\n")
    else:
        print("Датасет вже існує. Пропускаємо генерацію.\n")

# 3. Витягування ознак (Аналіз даних за допомогою MFCC)
def extract_mfcc(file_path, max_len=40):
    """Перетворює аудіосигнал у матрицю ознак MFCC."""
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T

def load_data():
    X, y = [], []
    for word in CLASSES:
        word_dir = os.path.join(DATASET_DIR, word)
        if not os.path.isdir(word_dir):
            continue
        for file in os.listdir(word_dir):
            if file.endswith(".wav") and not file.endswith("_base.wav"):
                file_path = os.path.join(word_dir, file)
                X.append(extract_mfcc(file_path))
                y.append(word)
    return np.array(X), np.array(y)

# 4. Основна логіка: Навчання та Тестування
def main():
    # Крок А: Підготовка даних
    generate_dataset(samples_per_word=10)
    print("Завантаження та обробка даних...")
    X, y_labels = load_data()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Крок Б: Створення та навчання нейромережі
    print("Створення та навчання нейромережі...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])), # (40, 20)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=25, batch_size=16,
              validation_data=(X_test, y_test), verbose=1)

    # Крок В: Симуляція розпізнавання бортового номера
    print("\n--- ТЕСТУВАННЯ: РОЗПІЗНАВАННЯ БОРТОВОГО НОМЕРА ---")
    test_sequence = ['bravo', 'seven', 'two']
    flight_number = ""

    print(f"Послідовність слів для розпізнавання: {test_sequence}")

    for word in test_sequence:
        # Обираємо випадковий файл з датасету
        random_index = random.randint(0, 9)
        test_file = os.path.join(DATASET_DIR, word, f"{word}_{random_index}.wav")

        print(f"  Аналізується файл: {test_file}")

        features = extract_mfcc(test_file)
        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features, verbose=0)
        predicted_class_index = np.argmax(prediction)
        predicted_word = le.inverse_transform([predicted_class_index])[0]

        symbol = AVIATION_ALPHABET[predicted_word]
        flight_number += symbol

        print(f"  Розпізнано слово: '{predicted_word}' -> Символ: '{symbol}'")

    print(f"\n[УСПІХ] Сформований бортовий номер літака: {flight_number}")

if __name__ == "__main__":
    main()