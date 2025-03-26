# utils/search.py
import asyncio
import logging
from typing import List
from duckduckgo_search import DDGS # Используем асинхронную версию

logger = logging.getLogger(__name__)

# Можно настроить User-Agent, если нужно
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

async def find_urls(query: str, num_results: int = 3) -> List[str]:
    """
    Ищет URL с использованием DuckDuckGo.

    Args:
        query: Поисковый запрос.
        num_results: Желаемое количество результатов.

    Returns:
        Список найденных URL.
    """
    logger.info(f"Выполнение поиска по запросу: '{query}', желаемое кол-во результатов: {num_results}")
    urls = []
    try:
        ddgs = DDGS(headers=HEADERS, timeout=10)
        results = ddgs.text(query, max_results=num_results * 2)
        
        for r in results:
            if r and isinstance(r, dict) and 'href' in r:
                urls.append(r['href'])
                if len(urls) >= num_results:
                    break
                    
        logger.info(f"Поиск по '{query}' вернул {len(urls)} URL.")

    except Exception as e:
        logger.exception(f"Ошибка при поиске DuckDuckGo для запроса '{query}': {e}")

    return urls