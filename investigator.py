import os
import telebot
import threading
import time
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
# 1. DUMMY WEB SERVER (FOR RENDER FREE TIER)
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
# 2. MAIN TELEGRAM BOT LOGIC
# ==========================================
load_dotenv()
bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))

# Global Expert Persona
GLOBAL_EXPERT_PROMPT = [
    "1. DYNAMIC PERSONA: Identify the field (Finance, Tech, etc.) and become the #1 world expert.",
    "2. DEEP RESEARCH: Synthesize the raw data provided to you.",
    "3. FORMAT: Use [Executive Summary], [Technical Deep-Dive], [Strategic Forecast], and [Sources].",
    "4. ACCURACY: Use strict financial terminology and logical deductions."
]

GITHUB_BASE_URL = "https://models.github.ai/inference"

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    # Initialize message ID tracker
    processing_msg = bot.reply_to(message, "🧠 AI Translator is reading your prompt...")

    try:
        # ------------------------------------------
        # STEP 0: THE TRANSLATOR (Groq Llama 3 8B)
        # ------------------------------------------
        translator_agent = Agent(
            model=Groq(id="llama-3.1-8b-instant"), 
            instructions=[
                "Translate casual user input into a highly specific, professional research directive.",
                "DO NOT answer. ONLY output the rewritten prompt."
            ]
        )
        super_prompt = translator_agent.run(message.text).content

        bot.edit_message_text(chat_id=message.chat.id, message_id=processing_msg.message_id, 
                              text="🤖 AI Orchestrator is selecting the best model...")

        # ------------------------------------------
        # STEP 0.5: THE ROUTER MANAGER (GitHub gpt-4o-mini)
        # ------------------------------------------
        router_agent = Agent(
            model=OpenAIChat(
                id="gpt-4o-mini", 
                api_key=os.getenv("GITHUB_TOKEN"),
                base_url=GITHUB_BASE_URL
            ),
            instructions=[
                "Choose the BEST model ID from this list only:",
                "- DeepSeek-R1 (Complex math/valuation/logic)",
                "- Cohere-command-r-plus (Massive news/text summaries)",
                "- o3-mini (Reasoning/strategy)",
                "- gpt-4o (Clean writing/formatting)",
                "- gpt-5-mini (Fast and smart)",
                "Output ONLY the ID."
            ]
        )
        
        chosen_model_id = router_agent.run(super_prompt).content.strip()
        
        # Valid List (Updated to match your Marketplace screenshots)
        valid_models = ["DeepSeek-R1", "Cohere-command-r-plus", "o3-mini", "gpt-4o", "gpt-5-mini"]
        if chosen_model_id not in valid_models:
            chosen_model_id = "gpt-4o-mini" # The ultimate safe default

        bot.edit_message_text(chat_id=message.chat.id, message_id=processing_msg.message_id, 
                              text=f"🕵️‍♂️ Gemini is scraping global data for {chosen_model_id}...")

        # ------------------------------------------
        # STEP 1: THE SCRAPER FALLBACK LOOP (Gemini Lite + Tools)
        # ------------------------------------------
        search_engines = [TavilyTools(), ExaTools(), DuckDuckGo()]
        scraped_facts = None
        
        for engine in search_engines:
            try:
                scraper_agent = Agent(
                    model=Gemini(id="gemini-2.5-flash-lite"),
                    tools=[engine, JinaReaderTools()],
                    instructions=["Find and dump all raw facts and numbers based on the prompt."]
                )
                raw_data_response = scraper_agent.run(super_prompt)
                scraped_facts = raw_data_response.content
                if scraped_facts: break
            except Exception as e:
                print(f"Engine failed: {e}")
                continue
        
        if not scraped_facts:
            raise Exception("No data could be retrieved by any search engine.")

        bot.edit_message_text(chat_id=message.chat.id, message_id=processing_msg.message_id, 
                              text=f"📝 {chosen_model_id} is analyzing and writing report...")

        # ------------------------------------------
        # STEP 2: THE DYNAMIC WRITER (GitHub Heavyweights)
        # ------------------------------------------
        writer_agent = Agent(
            model=OpenAIChat(
                id=chosen_model_id, 
                api_key=os.getenv("GITHUB_TOKEN"),
                base_url=GITHUB_BASE_URL
            ), 
            instructions=GLOBAL_EXPERT_PROMPT 
        )
        
        handoff_prompt = f"Data to analyze: {scraped_facts}\n\nTask: {super_prompt}"
        final_answer = writer_agent.run(handoff_prompt).content

        # ------------------------------------------
        # STEP 3: DELIVER TO TELEGRAM
        # ------------------------------------------
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, final_answer, parse_mode="Markdown")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}") # This will show up in Render Logs
        bot.edit_message_text(chat_id=message.chat.id, message_id=processing_msg.message_id,
                              text="🛑 The research network is busy. Please check your GitHub token or try again.")

# ==========================================
# 3. STARTUP SEQUENCE
# ==========================================
keep_alive()
print("🌍 ORCHESTRATOR IS ONLINE!")
bot.infinity_polling()
