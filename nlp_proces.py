import spacy
from spacy.lang.ru.stop_words import STOP_WORDS
import asyncpg
from dotenv import load_dotenv
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

load_dotenv()

class NLPProcessor:
    def __init__(self):
        self.nlp = spacy.load("ru_core_news_lg")
        self.STOP_WORDS = set(STOP_WORDS)
        self.db_pool = None
        self._vectorizer = None #векторизатор текста
        self.faq_questions = [] #список для хранения вопросов (FAQ)
        self.ai_assistant = None #хранит языковую модель
        
        #термины
        self.car_terms = {
            "сервис": ["техцентр", "сто", "ремонт", "сервисный"],
            "страховка": ["каско", "осаго", "автострахование"],
            "доставка": ["логистика", "перевозка", "транспортировка"],
            "hyundai": ["хендай", "хьюндай"],
            "creta": ["крета"],
            "solaris": ["солярис"],
            "дилерский центр": ["офис", "представительство", "дилер", "филиал"],
            "контакты": ["телефон", "адрес", "локация", "офис"]}
        
    #подключение к бд
    async def connect_db(self):
        self.db_pool = await asyncpg.create_pool(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"))
        
        await self.load_faq_data()

    #FAQ
    async def load_faq_data(self):
        async with self.db_pool.acquire() as conn:
            records = await conn.fetch("SELECT id, question FROM faq")
            self.faq_questions = [(rec['id'], rec['question']) for rec in records]
            
            questions_text = [q[1] for q in self.faq_questions]
            self._vectorizer = TfidfVectorizer()
            self._matrix = self._vectorizer.fit_transform(questions_text)

    #spaCy
    def preprocess_text(self, text):
        #нормализация номеров телефонов
        phone_ = r'(\d[\d\s\-\(\)]{5,}\d)'
        phones = re.findall(phone_, text)

        for phone in phones:
            clean_phone = re.sub(r'\D', '', phone)
            text = text.replace(phone, clean_phone)
        
        for term, synonyms in self.car_terms.items():
            if " " in term:
                for syn in synonyms:
                    text = text.replace(syn, term.replace(" ", "_"))

        #обработка текста spaСy
        doc = self.nlp(text.lower())
        lemmas = []

        for token in doc:
            #пропускаю пунктуацию, стоп слова и короткие слова
            if token.is_punct or token.is_stop or len(token.lemma_) < 2:
                continue
                
            #нормализация лемма
            lemma = token.lemma_.strip()
            
            #замена синонимов для тематики автомобилей
            for main_term, synonyms in self.car_terms.items():
                if lemma in synonyms:
                    lemma = main_term
                    break
                    
            if token.pos_ in ["NOUN", "ADJ", "PROPN"]:
                lemmas.append(lemma)
            
        return list(set(lemmas))
    
    #извлечение из сущностей
    def extract_entities(self, text):

        doc = self.nlp(text)
        entities = {"location": [], #локации
                    "service": [], #услуги
                    "contact": []} #контакты
        
        #распознование (NER)
        for ent in doc.ents:
            if ent.label_ in ["LOC", "GPE", "LOCATION"]:
                entities["location"].append(ent.text)
            elif ent.label_ in ["SERVICE", "CAR_MODEL"]:
                entities["service"].append(ent.text)
        
        #автомобильные тематики
        car_keywords = ["сервис", "страховк", "доставк", "логистик", "техосмотр"]
        
        for token in doc:
            lemma = token.lemma_.lower()

            if lemma in car_keywords:
                entities["service"].append(lemma)
        
        #поиск телефона
        phone_ = r'\b\d{7,11}\b'
        entities["contact"] = re.findall(phone_, text)
        
        return entities

    #поиск похожего FAQ
    async def match_faq(self, query):

        if not self._vectorizer:
            return None
            
        #преобразования запроса в вектор
        query_vec = self._vectorizer.transform([query])
        
        #расчет сходства
        similarities = cosine_similarity(query_vec, self._matrix)
        best_match_idx = similarities.argmax()
        best_similarity = similarities[0, best_match_idx]
        
        # Если сходство выше порога, возвращаем ответ
        if best_similarity > 0.3:

            faq_id, question = self.faq_questions[best_match_idx]

            async with self.db_pool.acquire() as conn:

                answer = await conn.fetchval("SELECT answer FROM faq WHERE id = $1", faq_id)
            
            return {
                "type": "faq",
                "answer": answer,
                "question": question,
                "confidence": float(best_similarity)}
        
        return None

    #поиск пододящей страницы
    async def match_page(self, keywords):

        if not keywords:
            return None

        async with self.db_pool.acquire() as conn:

            query = """
                    SELECT 
                    p.id, p.path, p.title, p.description,
                    COALESCE(SUM(k.weight), 0) AS total_weight
                    FROM pages p
                    JOIN keywords k ON p.id = k.page_id
                    WHERE k.keyword = ANY($1)
                    GROUP BY p.id
                    ORDER BY total_weight DESC
                    LIMIT 1"""
            
            records = await conn.fetch(query, keywords)
                
            if records and records[0]['total_weight'] > 0:
                best_match = records[0]
                
                # Простая нормализация confidence
                confidence = min(1.0, best_match['total_weight'] / 50.0)
                
                return {"type": "page",
                    "title": best_match['title'],
                    "description": best_match['description'],
                    "path": best_match['path'],
                    "confidence": round(confidence, 2)}
                
        return None

    #автоматический поиск подходящих ответов
    async def find_best_match(self, query):

        if not self.db_pool:
            await self.connect_db()
            
        #предобработка
        preprocessed_query = self.preprocess_text(query)
        query_ = " ".join(preprocessed_query)
        
        #поиск ответ в FAQ
        faq_match = await self.match_faq(query_)
        if faq_match:
            return faq_match
            
        #если FAQ не найдено ищет в страницах
        page_match = await self.match_page(preprocessed_query)
        if page_match:
            return page_match
            
        #если не найдено
        return {"type": "not_found", "answer": "Уточните ваш вопрос"}

    #интерфейс для общения с ИИ
    async def ask_assistant(self, query):
        if not self.ai_assistant:
            self.ai_assistant = AIAssistant(self)
        return await self.ai_assistant.generate(query)



class AIAssistant:
    def __init__(self, db_processor):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.db = db_processor
        self.system_prompt = """
        Ты ассистент компании Агат. Отвечай на вопросы клиентов.
        Отвечай на русском грамотно и вежливо. 
        Если информации нет в базе, предложи обратиться в поддержку."""

    #загрузка модели
    def load_model(self):

        if self.model is None:
            try:
                if self.device == "cuda":

                    bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="fp4",
                            bnb_4bit_compute_dtype=torch.float16)

                    self.model = AutoModelForCausalLM.from_pretrained(
                            "./ai_model",
                            device_map="auto",
                            quantization_config=bnb_config,
                            torch_dtype=torch.float16)
                else:

                    self.model = AutoModelForCausalLM.from_pretrained(
                            "./ai_model",
                            device_map="cpu",
                            torch_dtype=torch.float32)
                
                
                self.tokenizer = AutoTokenizer.from_pretrained("./ai_model")

            except Exception as e:
                print(f"Ошибка: {str(e)}")
                raise
        return self.model, self.tokenizer

    #генерация ответа
    async def generate(self, user_query):

        if len(user_query.strip()) < 2:
            return "Пожалуйста, уточните ваш вопрос."
        
        try:
            #поиск в БД
            db_match = await self.db.find_best_match(user_query)
            context = self.create_context(db_match)
            
            #формирование промпта
            prompt = f"{self.system_prompt}\nКонтекст: {context}\nВопрос: {user_query}\nОтвет:"
            
            #генерация ответа
            model, tokenizer = self.load_model()
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("Ответ:")[-1].strip()
        
        except Exception as e:
            print(f"Ошибка: {str(e)}")
            return "Извините, произошла ошибка. Попробуйте позже."

    #формирование контекста из результа
    def create_context(self, db_match):

        context = "Информация не найдена в базе знаний"

        if db_match.get("type") == "faq":
            context = f"FAQ: {db_match.get('question', '')} - {db_match.get('answer', '')}"
        elif db_match.get("type") == "page":
            context = f"Страница: {db_match.get('title', '')} ({db_match.get('path', '')}) - {db_match.get('description', '')}"
        return context