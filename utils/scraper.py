# utils/scraper.py
import asyncio
import logging
from typing import Dict, Optional
import httpx # Асинхронный HTTP клиент
from bs4 import BeautifulSoup # Для парсинга HTML

logger = logging.getLogger(__name__)

# Настройки для HTTP запросов
REQUEST_TIMEOUT = 15 # Секунды
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

async def scrape_content(url: str) -> Optional[Dict[str, str]]:
    """
    Асинхронно скачивает и извлекает основной текстовый контент со страницы.

    Args:
        url: URL страницы для скрапинга.

    Returns:
        Словарь {'url': url, 'text': extracted_text} или None в случае ошибки.
    """
    logger.info(f"Попытка скрапинга URL: {url}")
    try:
        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status() # Проверяем на ошибки HTTP (4xx, 5xx)

            content_type = response.headers.get("content-type", "").lower()
            if "html" not in content_type:
                logger.warning(f"Контент по URL {url} не является HTML (тип: {content_type}). Пропуск.")
                return None

            html_content = response.text
            soup = BeautifulSoup(html_content, 'lxml') # Используем lxml парсер, он быстрее

            # --- Стратегия извлечения текста ---
            # Удаляем ненужные теги (скрипты, стили, навигацию, футеры и т.д.)
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                element.decompose()

            # Извлекаем текст из основных тегов контента (простая эвристика)
            # Можно улучшить: искать тег <article>, <main>, или div с определенными классами
            body = soup.find('body')
            if not body:
                logger.warning(f"Не найден тег <body> на странице {url}. Пропуск.")
                return None

            # Собираем текст из параграфов, заголовков и списков внутри body
            text_parts = []
            for tag in body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                 # Получаем текст, убираем лишние пробелы и пустые строки
                cleaned_text = ' '.join(tag.get_text(separator=' ', strip=True).split())
                if cleaned_text:
                    text_parts.append(cleaned_text)

            extracted_text = "\n".join(text_parts)
            # ------------------------------------

            if not extracted_text:
                logger.warning(f"Не удалось извлечь текст со страницы {url} после парсинга.")
                return None

            logger.info(f"Успешно извлечено ~{len(extracted_text)} символов с {url}")
            return {"url": url, "text": extracted_text}

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP ошибка {e.response.status_code} при доступе к {url}: {e}")
        return None # Возвращаем None при ошибке HTTP
    except httpx.RequestError as e:
        # Включает таймауты, ошибки DNS, проблемы соединения
        logger.error(f"Ошибка запроса к {url}: {e}")
        return None
    except Exception as e:
        # Другие возможные ошибки (например, при парсинге BeautifulSoup)
        logger.exception(f"Неожиданная ошибка при скрапинге {url}: {e}")
        return None # Возвращаем None при любой другой ошибке