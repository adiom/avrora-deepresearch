# utils/llm.py
import google.generativeai as genai
import os
import logging
from typing import List, Dict, Any
import json # Для парсинга JSON ответов

logger = logging.getLogger(__name__)

# --- Настройка модели Gemini ---
# API ключ уже должен быть настроен в main.py
# Выбираем модель. 'gemini-1.5-flash' - быстрая и недорогая модель
# Если нужна более мощная, можно использовать 'gemini-1.5-pro'
MODEL_NAME = "gemini-1.5-flash"

# Настройки генерации (можно настроить)
generation_config = {
    "temperature": 0.7, # Контролирует случайность вывода
    "top_p": 1.0,
    "top_k": 32,
    "max_output_tokens": 8192, # Максимальное количество токенов в ответе
    # "response_mime_type": "application/json", # Если хотим строго JSON
}

safety_settings = [ # Настройки безопасности
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

try:
    # Используем системные инструкции для лучшего контроля над выводом
    system_instruction_search = """Ты - ИИ-ассистент для проведения исследований. Твоя задача - генерировать поисковые запросы."""
    model_search = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=system_instruction_search
    )

    system_instruction_analyze = """Ты - ИИ-аналитик исследований. Твоя задача - анализировать текст, извлекать ключевые выводы и предлагать новые направления для исследования. Отвечай ТОЛЬКО в формате JSON."""
    model_analyze = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"response_mime_type": "application/json", **generation_config}, # Требуем JSON
        safety_settings=safety_settings,
        system_instruction=system_instruction_analyze
    )

    system_instruction_report = """Ты - ИИ-писатель отчетов. Твоя задача - на основе предоставленных выводов составить подробный и структурированный отчет в формате Markdown."""
    model_report = genai.GenerativeModel(
        model_name=MODEL_NAME, # Для отчетов можно взять модель помощнее, если flash не справится
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=system_instruction_report
    )
    logger.info(f"Модели Gemini ({MODEL_NAME}) инициализированы.")

except Exception as e:
    logger.exception("Ошибка инициализации моделей Gemini!")
    # Это приведет к ошибке при запуске, но лучше знать сразу
    raise

async def _call_gemini_api(prompt: str, model: genai.GenerativeModel) -> str:
    """Вспомогательная функция для вызова API Gemini с обработкой ошибок."""
    logger.debug(f"Вызов Gemini API. Длина промпта: {len(prompt)} символов.")
    try:
        response = await model.generate_content_async(prompt)
        # Проверка на наличие текста в ответе
        if response.parts:
            result_text = response.text
            logger.debug(f"Gemini API ответил. Длина ответа: {len(result_text)} символов.")
            return result_text
        else:
            # Если Gemini заблокировал ответ из-за safety settings или по другой причине
            logger.warning(f"Gemini API вернул пустой ответ. Finish reason: {response.candidates[0].finish_reason}. Safety ratings: {response.candidates[0].safety_ratings}")
            return "" # Возвращаем пустую строку или можно кинуть ошибку
    except Exception as e:
        logger.exception(f"Ошибка при вызове Gemini API: {e}")
        raise # Передаем ошибку выше

async def generate_search_queries(context: str, learnings: List[str], breadth: int) -> List[str]:
    """Генерирует поисковые запросы с помощью Gemini."""
    prompt = f"""
Основываясь на следующем контексте исследования и уже полученных выводах, сгенерируй {breadth} новых, разнообразных и конкретных поисковых запросов для Google, чтобы углубить исследование.

**Контекст/Предыдущий запрос:**
{context}

**Уже известные выводы (последние 5):**
{learnings[-5:] if learnings else "Пока нет"}

**Задача:**
Сгенерируй ровно {breadth} поисковых запроса. Каждый запрос должен быть на новой строке, без нумерации или маркеров. Не добавляй никаких объяснений или вступлений, только сами запросы.

**Пример вывода:**
Лучшие практики квантовых вычислений 2024
Применение кубитов в медицине
Сравнение ионных ловушек и сверхпроводящих кубитов
"""
    response_text = await _call_gemini_api(prompt, model_search)
    queries = [q.strip() for q in response_text.split('\n') if q.strip()]
    logger.info(f"Сгенерированные запросы из ответа Gemini: {queries}")
    # Возьмем не больше breadth запросов, если модель дала больше
    return queries[:breadth]

async def summarize_and_find_directions(context: str, learnings: List[str], text_to_analyze: str) -> Dict[str, List[str]]:
    """Анализирует текст, извлекает выводы и направления в формате JSON."""
    # Ограничим размер текста, чтобы не превышать лимиты (очень грубое ограничение)
    max_input_chars = 30000 # Примерный лимит, нужно смотреть актуальные для модели
    if len(text_to_analyze) > max_input_chars:
        logger.warning(f"Текст для анализа ({len(text_to_analyze)} символов) слишком большой, обрезается до {max_input_chars}.")
        text_to_analyze = text_to_analyze[:max_input_chars]

    prompt = f"""
Проанализируй следующий текст, полученный в ходе веб-исследования.
Учитывай первоначальный запрос и уже известные выводы.

**Первоначальный запрос/Контекст:**
{context}

**Уже известные выводы:**
{learnings if learnings else "Пока нет"}

**Текст для анализа (из нескольких источников):**

{text_to_analyze}


**Твоя задача:**
Верни ТОЛЬКО JSON объект со следующей структурой:
{{
  "learnings": [
    "Краткий вывод 1 на основе текста",
    "Краткий вывод 2, который является новым знанием",
    "..."
  ],
  "directions": [
    "Конкретный вопрос для дальнейшего исследования, возникший из текста",
    "Направление для углубления, связанное с темой",
    "..."
  ]
}}
Избегай дублирования выводов, которые уже есть в `learnings`. Сосредоточься на новой информации из `Текст для анализа`.
Выводы (`learnings`) должны быть краткими утверждениями.
Направления (`directions`) должны быть вопросами или темами для дальнейшего поиска.
Если текст не содержит полезной информации, верни пустые списки.
"""
    response_text = await _call_gemini_api(prompt, model_analyze)

    try:
        # Попытка распарсить JSON
        result_json = json.loads(response_text)
        # Проверка структуры
        if isinstance(result_json, dict) and \
           isinstance(result_json.get("learnings"), list) and \
           isinstance(result_json.get("directions"), list):
            logger.info(f"Успешно извлечены выводы: {result_json.get('learnings')}, направления: {result_json.get('directions')}")
            return result_json
        else:
            logger.warning(f"Gemini вернул некорректный JSON: {response_text}")
            return {"learnings": [], "directions": []}
    except json.JSONDecodeError:
        logger.error(f"Не удалось распарсить JSON ответ от Gemini: {response_text}")
        # Попытка извлечь хоть что-то (очень грубо)
        learnings_guess = []
        directions_guess = []
        if '"learnings": [' in response_text:
            # Простая попытка извлечь строки из списка learnings
            pass # Добавить логику извлечения если нужно
        if '"directions": [' in response_text:
             # Простая попытка извлечь строки из списка directions
            pass # Добавить логику извлечения если нужно
        return {"learnings": learnings_guess, "directions": directions_guess}
    except Exception as e:
        logger.exception(f"Неожиданная ошибка при обработке ответа Gemini: {e}")
        return {"learnings": [], "directions": []}

async def generate_final_report(query: str, all_learnings: List[str]) -> str:
    """Генерирует итоговый отчет в Markdown."""
    if not all_learnings:
        logger.warning("Нет выводов для генерации отчета.")
        return f"# Исследование по запросу: {query}\n\nНе удалось собрать информацию по данному запросу."

    prompt = f"""
Создай подробный и структурированный итоговый отчет в формате Markdown по результатам исследования.

**Первоначальный запрос:**
{query}

**Ключевые выводы, собранные в ходе исследования:**

{all_learnings}


**Задача:**
Напиши отчет, который:
1.  Начинается с заголовка `# Исследование по запросу: {query}`.
2.  Содержит краткое введение (1-2 предложения), суммирующее цель исследования.
3.  Представляет основные выводы в логически структурированном виде. Используй подзаголовки (`##`), списки (`*` или `-`), выделение (`**жирным**`).
4.  Группируй связанные выводы вместе.
5.  Не просто перечисляй выводы, а связывай их в единый текст.
6.  Заканчивается разделом `## Заключение`, где подводится итог исследования (1-2 предложения).
7.  Не включай список источников, он будет добавлен отдельно.
8.  Сделай отчет понятным и информативным.
"""
    report = await _call_gemini_api(prompt, model_report)
    return report if report else f"# Отчет не сгенерирован\n\nGemini не вернул текст отчета.\n\nСобранные выводы:\n" + "\n".join(f"- {l}" for l in all_learnings)