from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
import os
import httpx
import logging
from telegram import Update
from telegram.ext import (
    Application, ContextTypes, MessageHandler, filters
)
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate required environment variables
if not all([TELEGRAM_TOKEN, BRAVE_API_KEY, OPENAI_API_KEY]):
    raise ValueError("Missing required environment variables")

telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()

# Global HTTP client for reuse
http_client = None

# 🔁 FastAPI Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=30.0)
    
    await telegram_app.initialize()
    await telegram_app.start()
    logger.info("Telegram bot started")
    
    yield
    
    await telegram_app.stop()
    await http_client.aclose()
    logger.info("Application stopped")

app = FastAPI(lifespan=lifespan)

# Brave Search with error handling
async def brave_search(query: str) -> str:
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    params = {"q": query, "count": 3}
    
    try:
        resp = await http_client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        if "web" in data and "results" in data["web"] and data["web"]["results"]:
            results = data["web"]["results"]
            return "\n".join([
                f"{r.get('title', 'No title')} - {r.get('url', 'No URL')}" 
                for r in results[:3]
            ])
        return "No search results found."
        
    except httpx.HTTPError as e:
        logger.error(f"Brave search error: {e}")
        return "Search service temporarily unavailable."
    except Exception as e:
        logger.error(f"Unexpected error in brave_search: {e}")
        return "An error occurred during search."

# GPT Summary with error handling
async def summarize_with_openai(text: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful assistant that summarizes search results concisely."
            },
            {
                "role": "user", 
                "content": f"Summarize and explain these search results:\n{text}"
            }
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        response = await http_client.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"]
        return "Unable to generate summary."
        
    except httpx.HTTPError as e:
        logger.error(f"OpenAI API error: {e}")
        return "Summary service temporarily unavailable."
    except Exception as e:
        logger.error(f"Unexpected error in summarize_with_openai: {e}")
        return "An error occurred while generating summary."

# Telegram Handler with error handling
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not update.message.text:
            return
            
        query = update.message.text.strip()
        if not query:
            await update.message.reply_text("Please send a search query.")
            return
            
        # Search
        await update.message.reply_text("🔎 Searching...")
        results = await brave_search(query)
        
        if "error" in results.lower() or "unavailable" in results.lower():
            await update.message.reply_text(results)
            return
            
        # Summarize
        await update.message.reply_text("💡 Generating summary...")
        summary = await summarize_with_openai(results)
        
        # Send result
        await update.message.reply_text(summary)
        
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await update.message.reply_text("Sorry, an error occurred. Please try again.")

telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Webhook Endpoint with basic validation
@app.post("/webhook")
async def telegram_webhook(request: Request):
    try:
        json_data = await request.json()
        update = Update.de_json(json_data, telegram_app.bot)
        
        if update:
            await telegram_app.process_update(update)
            return {"ok": True}
        else:
            raise HTTPException(status_code=400, detail="Invalid update data")
            
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "bot_running": telegram_app.running}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)