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
GITHUB_BASE_URL = "https://models.inference.ai.azure.com"

GLOBAL_EXPERT_PROMPT = [
    "1. PERSONA: You are a Tier-1 Investment Analyst.",
    "2. TASK: Synthesize provided raw data into a professional report.",
    "3. FORMAT: Use [Executive Summary], [Technical Deep-Dive], [Strategic Forecast].",
    "4. LANGUAGE: Professional, concise, and data-driven."
]

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
            instructions=["Pick one: DeepSeek-R1, Cohere-command-r-plus, o3-mini, Meta-Llama-3.1-405B-Instruct. Output ONLY the ID."]
        )
        chosen_id = router.run(super_prompt).content.strip()
        if chosen_id not in ["DeepSeek-R1", "Cohere-command-r-plus", "o3-mini", "Meta-Llama-3.1-405B-Instruct"]: chosen_id = "o3-mini"

        # STEP 1: SELF-HEALING SCRAPER
        bot.edit_message_text(f"🕵️‍♂️ Scraper: Gathering data for {chosen_id}...", message.chat.id, processing_msg.message_id)
        
        scraped_data = None
        try:
            # Primary: Gemini
            scraper = Agent(
                model=Gemini(id="gemini-2.5-flash-lite"),
                tools=[TavilyTools(), JinaReaderTools()],
                instructions=["Extract raw facts and financial data."]
            )
            scraped_data = scraper.run(super_prompt).content
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                bot.edit_message_text("🔄 Gemini exhausted. Switching to Groq Scraper...", message.chat.id, processing_msg.message_id)
                # Fallback: Groq
                scraper = Agent(
                    model=Groq(id="llama-3.3-70b-versatile"),
                    tools=[ExaTools(), DuckDuckGo()],
                    instructions=["Extract raw facts and financial data."]
                )
                scraped_data = scraper.run(super_prompt).content
            else:
                raise e

        # STEP 2: WRITER (GitHub)
        bot.edit_message_text(f"📝 {chosen_id}: Writing report...", message.chat.id, processing_msg.message_id)
        writer = Agent(
            model=OpenAIChat(id=chosen_id, api_key=os.getenv("GITHUB_TOKEN"), base_url=GITHUB_BASE_URL),
            instructions=GLOBAL_EXPERT_PROMPT
        )
        final_answer = writer.run(f"Context: {scraped_data}\n\nTask: {super_prompt}").content

        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, final_answer, parse_mode="Markdown")

    except Exception as e:
        bot.edit_message_text(f"🛑 Error: {type(e).__name__}\nDetail: {str(e)[:100]}", message.chat.id, processing_msg.message_id)

# ==========================================
# 3. START
# ==========================================
keep_alive()
bot.remove_webhook()
print("🌍 ORCHESTRATOR IS ONLINE!")
bot.infinity_polling()
