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
# Stable endpoint for 2026
GH_URL = "https://models.inference.ai.azure.com"

# --- RENDER KEEP-ALIVE ---
def run_s(): 
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), type('H', (BaseHTTPRequestHandler,), {'do_GET': lambda s: (s.send_response(200), s.end_headers(), s.wfile.write(b'LIVE'))}))
    server.serve_forever()
threading.Thread(target=run_s, daemon=True).start()

@bot.message_handler(func=lambda m: True)
def handle(m):
    p_msg = bot.reply_to(m, "🔋 Expert Research System: Online")
    try:
        # 1. OPTIMIZE PROMPT (Groq)
        bot.edit_message_text("🧠 Thinking: Refining query...", m.chat.id, p_msg.message_id)
        super_prompt = Agent(model=Groq(id="llama-3.1-8b-instant")).run(f"Rewrite into expert directive: {m.text}").content

        # 2. DATA SCRAPING (Gemini - No Jina)
        bot.edit_message_text("🔍 Searching: Gathering market data...", m.chat.id, p_msg.message_id)
        try:
            # Tavily gives clean, concise snippets (No 413 error)
            raw_data = Agent(model=Gemini(id="gemini-2.5-flash-lite"), tools=[TavilyTools()]).run(super_prompt).content
        except:
            # Fallback to DuckDuckGo if Gemini/Tavily is out of quota
            raw_data = Agent(model=Groq(id="llama-3.3-70b-versatile"), tools=[DuckDuckGo()]).run(super_prompt).content
        
        # FINAL PROTECTION: Strictly limit text size to 8,000 chars for the writer
        clean_data = (raw_data[:8000] + "..[TRUNCATED]") if len(raw_data) > 8000 else raw_data

        # 3. EXPERT WRITER (Double Failover for 400 Error)
        bot.edit_message_text("📝 Writing: Drafting professional report...", m.chat.id, p_msg.message_id)
        
        # We TRY o3-mini first. If it's "Unavailable" (400), we jump to gpt-4o-mini.
        target_model = "o3-mini"
        try:
            writer_agent = Agent(
                model=OpenAIChat(id=target_model, api_key=os.getenv("GITHUB_TOKEN"), base_url=GH_URL),
                instructions=["Tier-1 Analyst style. 3 Bullets. 1 Data Table. 1 Verdict."]
            )
            report = writer_agent.run(f"Data: {clean_data}\n\nTask: {super_prompt}").content
        except Exception as e:
            if "400" in str(e) or "unavailable" in str(e).lower():
                # FAILOVER TO STABLE MODEL
                target_model = "gpt-4o-mini"
                writer_agent = Agent(
                    model=OpenAIChat(id=target_model, api_key=os.getenv("GITHUB_TOKEN"), base_url=GH_URL),
                    instructions=["Tier-1 Analyst style. Concise. 3 Bullets. 1 Table."]
                )
                report = writer_agent.run(f"Data: {clean_data}\n\nTask: {super_prompt}").content
            else:
                raise e

        bot.delete_message(m.chat.id, p_msg.message_id)
        bot.send_message(m.chat.id, report, parse_mode="Markdown")

    except Exception as e:
        bot.edit_message_text(f"🛑 Network Error: {str(e)[:50]}", m.chat.id, p_msg.message_id)

# Kill old sessions and start
bot.remove_webhook()
bot.infinity_polling()
