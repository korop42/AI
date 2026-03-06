from google import genai

# 1. Налаштування клієнта (вставте сюди свій ключ)
API_KEY = ''
client = genai.Client(api_key=API_KEY)

# 2. Створення сесії чату з актуальною моделлю
# Використовуємо найновішу швидку модель Flash
chat = client.chats.create(
    model="gemini-2.5-flash",
    config={

    }
)

print("="*50)
print("🤖 Чат-бот (Нова версія) запущено! (Напишіть 'вихід' для завершення)")
print("="*50)

# 3. Головний цикл програми
while True:
    # Отримуємо запит від користувача
    user_input = input("\nВи: ")

    # Умова виходу з програми
    if user_input.lower() in ['вихід', 'exit', 'quit', 'q']:
        print("🤖 Бот: До зустрічі! Роботу завершено.")
        break

    # Якщо користувач нічого не ввів, ігноруємо
    if not user_input.strip():
        continue

    try:
        # Відправляємо запит через новий клієнт та отримуємо відповідь
        response = chat.send_message(user_input)
        print(f"🤖 Бот: {response.text}")
    except Exception as e:
        print(f"❌ Виникла помилка: {e}")