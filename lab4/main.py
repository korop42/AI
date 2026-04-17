import streamlit as st
from gtts import gTTS
import os
import uuid

# ===== НАЛАШТУВАННЯ СТОРІНКИ =====
st.set_page_config(
    page_title="Speech Generator",
    page_icon="🎤",
    layout="centered"
)

st.header("🎤 Text → Speech (Lab 4)")
st.caption("Введіть текст і отримайте озвучку")

# ===== ВВІД КОРИСТУВАЧА =====
user_text = st.text_area(
    label="Текст для озвучення:",
    placeholder="Наприклад: Привіт, як справи?",
    height=140
)

# ===== ВИБІР МОВИ =====
language_options = {
    "Українська": "uk",
    "English": "en",
    "Deutsch": "de"
}

selected_lang_name = st.selectbox("Мова:", list(language_options.keys()))
selected_lang_code = language_options[selected_lang_name]

# ===== ОСНОВНА ЛОГІКА =====
def generate_audio(text, lang_code):
    unique_name = f"audio_{uuid.uuid4().hex}.mp3"

    tts = gTTS(text=text, lang=lang_code)
    tts.save(unique_name)

    return unique_name


# ===== КНОПКА =====
if st.button("▶️ Озвучити текст"):
    if not user_text.strip():
        st.warning("⚠️ Введіть текст перед генерацією")
    else:
        try:
            with st.spinner("⏳ Обробка..."):
                audio_file = generate_audio(user_text, selected_lang_code)

                st.success("✅ Готово!")

                # Відтворення
                st.audio(audio_file)

                # Завантаження
                with open(audio_file, "rb") as f:
                    st.download_button(
                        label="⬇️ Завантажити",
                        data=f,
                        file_name="result.mp3",
                        mime="audio/mp3"
                    )

                # Видалення
                os.remove(audio_file)

        except Exception as error:
            st.error(f"❌ Помилка: {error}")