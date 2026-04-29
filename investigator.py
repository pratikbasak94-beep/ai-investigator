import os, telebot, threading, requests
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.tavily import TavilyTools
from phi.tools.duckduckgo import DuckDuckGo
from http.server import BaseHTTPRequestHandler, HTTPServer

load_dotenv()
bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))
GH_URL = "https://models.inference.ai.azure.com"

# --- RENDER KEEP-ALIVE ---
def run_s(): 
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), type('H', (BaseHTTPRequestHandler,), {'do_GET': lambda s: (s.send_response(200), s.end_headers(), s.wfile.write(b'STAY_ALIVE'))}))
    server.serve_forever()
threading.Thread(target=run_s, daemon=True).start()

@bot.message_handler(func=lambda m: True)
def handle(m):
    p_msg = bot.reply_to(m, "🕵️‍♂️ Expert Brain Initializing...")
    try:
        # 1. SMART PROMPT OPTIMIZER (Groq)
        bot.edit_message_text("🧠 Optimizing Research Query...", m.chat.id, p_msg.message_id)
        prompt_agent = Agent(model=Groq(id="llama-3.1-8b-instant"))
        super_prompt = prompt_agent.run(f"Convert to expert research directive: {m.text}").content

        # 2. SELECT BEST CLOUD WRITER (GitHub)
        router = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("GITHUB_TOKEN"), base_url=GH_URL),
            instructions=["Pick either 'DeepSeek-R1' (Math/Logic) or 'o3-mini' (News/Strategy). Output ID only."]
        )
        chosen_id = router.run(super_prompt).content.strip()
        if chosen_id not in ["DeepSeek-R1", "o3-mini"]: chosen_id = "o3-mini"

        # 3. DATA SCRAPER (Gemini with Groq Fallback) - NO JINA
        bot.edit_message_text(f"🔍 Searching for live data (using {chosen_id})...", m.chat.id, p_msg.message_id)
        
        scraped_data = ""
        try:
            # Primary Scraper: Tavily (Fastest for Finance)
            scraper = Agent(model=Gemini(id="gemini-2.5-flash-lite"), tools=[TavilyTools()])
            scraped_data = scraper.run(super_prompt).content
        except:
            # Backup Scraper: DuckDuckGo (Lightweight)
            scraper = Agent(model=Groq(id="llama-3.3-70b-versatile"), tools=[DuckDuckGo()])
            scraped_data = scraper.run(super_prompt).content
        
        # PROTECTION: Truncate data to avoid 413 error
        # Cuts data to roughly 10,000 characters to stay safe
        scraped_data = (scraped_data[:10000] + "..[Truncated]") if len(scraped_data) > 10000 else scraped_data

        # 4. FINAL EXPERT REPORT
        bot.edit_message_text(f"📝 {chosen_id} is generating report...", m.chat.id, p_msg.message_id)
        
        writer = Agent(
            model=OpenAIChat(id=chosen_id, api_key=os.getenv("GITHUB_TOKEN"), base_url=GH_URL),
            instructions=[
                "Format: 1. EXECUTIVE SUMMARY (3 Bullet Points)",
                "2. DATA TABLE (Key Metrics/Stocks)",
                "3. FINAL VERDICT (Buy/Sell/Watch)",
                "Keep the tone professional and the output concise."
            ]
        )
        
        final_report = writer.run(f"Data: {scraped_data}\n\nTask: {super_prompt}").content

        bot.delete_message(m.chat.id, p_msg.message_id)
        bot.send_message(m.chat.id, final_report, parse_mode="Markdown")

    except Exception as e:
        bot.edit_message_text(f"🛑 Error: {str(e)[:50]}", m.chat.id, p_msg.message_id)

# Start Polling
bot.remove_webhook()
bot.infinity_polling()
