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
# 1. DUMMY WEB SERVER (FOR RENDER FREE TIER)
# ==========================================
class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Bot is alive and running on Render!")

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

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    processing_msg = bot.reply_to(message, "🧠 AI Translator is reading your prompt...")

    try:
        # ------------------------------------------
        # STEP 0: THE TRANSLATOR (Groq Llama 3 8B)
        # ------------------------------------------
        translator_agent = Agent(
            model=Groq(id="llama-3.1-8b-instant"), 
            instructions=[
                "You are an expert Prompt Engineer.",
                "Take the user's casual message and rewrite it into a highly specific, professional directive for a research AI.",
                "Identify missing keywords (like adding 'stock' or 'India' if implied).",
                "DO NOT answer the user's question. ONLY output the rewritten prompt."
            ]
        )
        super_prompt = translator_agent.run(message.text).content

        bot.edit_message_text(chat_id=message.chat.id, message_id=processing_msg.message_id, 
                              text="🤖 AI Orchestrator is routing to the best expert model...")

        # ------------------------------------------
        # STEP 0.5: THE ROUTER MANAGER (GitHub gpt-4o-mini)
        # ------------------------------------------
        router_agent = Agent(
            model=OpenAIChat(
                id="gpt-4o-mini", 
                api_key=os.getenv("GITHUB_TOKEN"),
                base_url="https://models.inference.ai.azure.com"
            ),
            instructions=[
                "You are an AI Orchestrator.",
                "Review the user's research request and select the BEST model to write the final report.",
                "Choose from this list strictly based on the model's strength:",
                "- DeepSeek-R1 : Use for heavy math, stock valuation, and deep financial logic.",
                "- Cohere-command-r-plus-08-2024 : Use for synthesizing massive amounts of news or market data.",
                "- o3-mini : Use for strategic forecasting, predictions, and complex planning.",
                "- Meta-Llama-3.1-405B-Instruct : Use for broad industry overviews and macro-economic research.",
                "- gpt-4o : Use for general formatting, clean writing, and standard queries.",
                "DO NOT output anything except the exact text of the model ID from the list."
            ]
        )
        
        chosen_model_id = router_agent.run(super_prompt).content.strip()
        
        # Expanded Failsafe List
        valid_models = [
            "DeepSeek-R1", 
            "Cohere-command-r-plus-08-2024", 
            "o3-mini", 
            "Meta-Llama-3.1-405B-Instruct", 
            "gpt-4o"
        ]
        
        # If the router hallucinates, default to GPT-4o
        if chosen_model_id not in valid_models:
            chosen_model_id = "gpt-4o"

        bot.edit_message_text(chat_id=message.chat.id, message_id=processing_msg.message_id, 
                              text=f"🕵️‍♂️ Gemini Lite is scraping global data for {chosen_model_id}...")

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
                    instructions=[
                        "You are a backend data scraper.",
                        "Use your tools to find as much raw data and facts as possible based on the prompt.",
                        "Do not format the output nicely. Just dump the raw facts and numbers."
                    ]
                )
                raw_data_response = scraper_agent.run(super_prompt)
                scraped_facts = raw_data_response.content
                break  # Successful! Break the loop.
            except Exception:
                continue  # If Tavily/Jina hits a limit, try the next engine
        
        if not scraped_facts:
            raise Exception("All search engines failed.")

        bot.edit_message_text(chat_id=message.chat.id, message_id=processing_msg.message_id, 
                              text=f"📝 {chosen_model_id} is analyzing data and writing the report...")

        # ------------------------------------------
        # STEP 2: THE DYNAMIC WRITER (GitHub Heavyweights)
        # ------------------------------------------
        writer_agent = Agent(
            model=OpenAIChat(
                id=chosen_model_id, 
                api_key=os.getenv("GITHUB_TOKEN"),
                base_url="https://models.inference.ai.azure.com"
            ), 
            instructions=GLOBAL_EXPERT_PROMPT 
        )
        
        handoff_prompt = f"""
        Original Task: {super_prompt}
        
        Raw Data Found by Research Team:
        {scraped_facts}
        
        Task: Write the final report answering the task using ONLY the raw data provided above.
        """
        
        final_report_response = writer_agent.run(handoff_prompt)
        final_answer = final_report_response.content

        # ------------------------------------------
        # STEP 3: DELIVER TO TELEGRAM
        # ------------------------------------------
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, final_answer)

    except Exception as e:
        bot.edit_message_text(chat_id=message.chat.id, message_id=processing_msg.message_id,
                              text="🛑 The global research network is currently busy. Please try again in a few moments.")
        print(f"Error: {e}")

# ==========================================
# 3. STARTUP SEQUENCE
# ==========================================
keep_alive()

print("🌍 GLOBAL ORCHESTRATOR SYSTEM IS LIVE!")
bot.infinity_polling()
