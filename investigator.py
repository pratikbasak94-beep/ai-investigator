import os
import telebot
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.tavily import TavilyTools
from phi.tools.exa import ExaTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.jina_tools import JinaReaderTools

# ==========================================
# 1. DUMMY WEB SERVER
# ==========================================
class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Investigator AI is alive!")

def run_dummy_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), DummyHandler)
    server.serve_forever()

def keep_alive():
    t = threading.Thread(target=run_dummy_server)
    t.daemon = True
    t.start()

# ==========================================
# 2. MAIN BOT LOGIC
# ==========================================
load_dotenv()
bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))

# Using the standard Azure-style URL which GitHub Models requires
GITHUB_BASE_URL = "https://models.inference.ai.azure.com"

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    processing_msg = bot.reply_to(message, "🧠 AI Manager: Starting analysis...")

    try:
        # STEP 0: TRANSLATOR (Groq)
        bot.edit_message_text("🧠 Translator: Optimizing prompt...", message.chat.id, processing_msg.message_id)
        translator = Agent(model=Groq(id="llama-3.1-8b-instant"), instructions=["Rewrite as a pro research prompt. Output ONLY the rewrite."])
        super_prompt = translator.run(message.text).content

        # STEP 0.5: ROUTER (GitHub)
        bot.edit_message_text("🤖 Orchestrator: Routing to expert...", message.chat.id, processing_msg.message_id)
        router = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("GITHUB_TOKEN"), base_url=GITHUB_BASE_URL),
            instructions=["Pick one ID: DeepSeek-R1, Cohere-command-r-plus, o3-mini, gpt-4o. Output ONLY the ID."]
        )
        chosen_id = router.run(super_prompt).content.strip()
        
        # Valid ID check
        valid = ["DeepSeek-R1", "Cohere-command-r-plus", "o3-mini", "gpt-4o"]
        if chosen_id not in valid: chosen_id = "gpt-4o"

        # STEP 1: SCRAPER (Gemini)
        bot.edit_message_text(f"🕵️‍♂️ Gemini: Scraping data for {chosen_id}...", message.chat.id, processing_msg.message_id)
        scraper = Agent(
            model=Gemini(id="gemini-2.5-flash-lite"),
            tools=[TavilyTools(), ExaTools(), JinaReaderTools()],
            instructions=["Dump raw data and facts based on the prompt."]
        )
        scraped_data = scraper.run(super_prompt).content

        # STEP 2: WRITER (GitHub)
        bot.edit_message_text(f"📝 {chosen_id}: Writing report...", message.chat.id, processing_msg.message_id)
        writer = Agent(
            model=OpenAIChat(id=chosen_id, api_key=os.getenv("GITHUB_TOKEN"), base_url=GITHUB_BASE_URL),
            instructions=["Use [Executive Summary] and [Strategic Forecast]."]
        )
        final_answer = writer.run(f"Context: {scraped_data}\n\nTask: {super_prompt}").content

        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, final_answer, parse_mode="Markdown")

    except Exception as e:
        # THE DEBUG FIX: This will tell us the exact error in Telegram
        error_type = type(e).__name__
        bot.edit_message_text(f"🛑 Error: {error_type}\nDetail: {str(e)[:100]}", message.chat.id, processing_msg.message_id)

# ==========================================
# 3. START
# ==========================================
keep_alive()
print("🌍 BOT STARTED")
bot.infinity_polling()
