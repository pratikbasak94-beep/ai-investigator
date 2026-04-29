import os
import telebot
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.model.groq import Groq
from phi.tools.tavily import TavilyTools
from phi.tools.exa import ExaTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.jina_tools import JinaReaderTools

# ==========================================
# 1. DUMMY WEB SERVER (FOR RENDER FREE TIER)
# ==========================================
class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Bot is alive and running on Render!")

def run_dummy_server():
    # Render assigns a port automatically. If running locally, defaults to 10000.
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), DummyHandler)
    server.serve_forever()

def keep_alive():
    # Start the web server in a separate background thread
    t = threading.Thread(target=run_dummy_server)
    t.daemon = True
    t.start()

# ==========================================
# 2. MAIN TELEGRAM BOT LOGIC
# ==========================================

# Load keys
load_dotenv()
bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))

# Global Expert Persona
GLOBAL_EXPERT_PROMPT = [
    "1. DYNAMIC PERSONA: Identify the field (Finance, Tech, etc.) and become the #1 world expert.",
    "2. DEEP RESEARCH: Use Tavily/Exa/Jina to provide high-density data and specific sources.",
    "3. FORMAT: Use [Executive Summary], [Technical Deep-Dive], [Strategic Forecast], and [Sources].",
    "4. ACCURACY: Use LaTeX for any math or financial formulas."
]

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    # Tiered Search Engines
    search_engines = [TavilyTools(), ExaTools(), DuckDuckGo()]
    
    # Tiered AI Brains (2026 Model IDs)
    ai_models = [
        ("Gemini Pro", Gemini(id="gemini-2.5-flash")),
        ("Gemini Lite", Gemini(id="gemini-2.5-flash-lite")),
        ("Groq Llama", Groq(id="llama-3.3-70b-versatile"))
    ]

    for engine in search_engines:
        try:
            agent = Agent(
                tools=[engine, JinaReaderTools()],
                instructions=GLOBAL_EXPERT_PROMPT,
                markdown=True
            )

            for provider, model_instance in ai_models:
                try:
                    agent.model = model_instance
                    bot.reply_to(message, f"📡 {provider} Expert scanning via {type(engine).__name__}...")
                    
                    response = agent.run(message.text)
                    bot.send_message(message.chat.id, response.content)
                    return 
                except Exception:
                    continue # Try next brain
        except Exception:
            continue # Try next search engine

    bot.send_message(message.chat.id, "🛑 All AI systems are busy. Try again in a moment.")

# ==========================================
# 3. STARTUP SEQUENCE
# ==========================================

# Start the dummy web server so Render doesn't crash
keep_alive()

print("🌍 GLOBAL EXPERT SYSTEM IS LIVE (2026 EDITION)!")
# Start the bot polling
bot.infinity_polling()
