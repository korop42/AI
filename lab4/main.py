import streamlit as st
from gtts import gTTS
import os

# Налаштування сторінки
st.set_page_config(page_title="TTS Додаток", page_icon="🎙️")

st.title("🎙️ Генератор мовлення (Лабораторна №4)")
st.write("Введіть текст нижче, щоб перетворити його на аудіо.")

# Форма для введення тексту
text_input = st.text_area("Ваш текст:", placeholder="Привіт! Як справи?", height=150)

# Вибір мови
lang = st.selectbox("Оберіть мову:", [
    ("Українська", "uk"),
    ("English", "en"),
    ("Deutsch", "de")
], format_func=lambda x: x[0])

if st.button("Згенерувати мовлення"):
    if text_input.strip():
        try:
            with st.spinner('Генеруємо аудіо...'):
                # Використання gTTS API
                tts = gTTS(text=text_input, lang=lang[1])
                filename = "speech.mp3"
                tts.save(filename)

                # Відтворення в браузері
                st.audio(filename, format="audio/mp3")

                # Кнопка для завантаження файлу
                with open(filename, "rb") as file:
                    st.download_button(
                        label="📥 Завантажити MP3",
                        data=file,
                        file_name="speech.mp3",
                        mime="audio/mp3"
                    )

                # Очищення тимчасового файлу
                os.remove(filename)
        except Exception as e:
            st.error(f"❌ Сталася помилка: {e}")
    else:
        st.warning("⚠️ Будь ласка, введіть текст.")