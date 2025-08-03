import asyncio
from nlp_proces import NLPProcessor


async def main():
    processor = NLPProcessor()
    await processor.connect_db()
    
    print("Задайте вопрос:")
    
    while True:
        try:
            user_input = input("Вы: ")
            if user_input.lower() in ['выход']:
                break
                
            response = await processor.ask_assistant(user_input)
            print(f"\nАссистент: {response}\n")

        except KeyboardInterrupt:
            break

        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())