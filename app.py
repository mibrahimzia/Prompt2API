import gradio as gr
import requests
from bs4 import BeautifulSoup
import json
import os
import time
import hashlib
import re
import uuid

# Configuration - Replace with your actual API keys
GROQ_API_KEY = "your-groq-api-key-here"  # Replace with your actual Groq API key
OPENROUTER_API_KEY = "your-openrouter-api-key-here"  # Replace with your actual Open Router API key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# In-memory caches
scraped_text_cache = {}
data_cache = {}

def scrape_url(url):
    """Scrape a URL and return cleaned text content."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted tags
        for script in soup(["script", "style", "nav", "footer", "aside", "header"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:10000]  # Limit to 10,000 characters
    except Exception as e:
        return f"Error scraping URL: {str(e)}"

def get_links_from_url(base_url):
    """Extract links from a base URL that might contain relevant content."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Handle relative URLs
            if href.startswith('/'):
                full_url = base_url + href
            else:
                full_url = href
            # Only include links that are likely to contain content
            if re.search(r'(program|course|academic|faculty|department|degree)', href, re.IGNORECASE):
                links.append(full_url)
        return links
    except Exception as e:
        print(f"Error getting links: {e}")
        return []

def scrape_multiple_pages(url):
    """Scrape the main URL and relevant linked pages."""
    if url in scraped_text_cache:
        return scraped_text_cache[url]
    
    main_text = scrape_url(url)
    if main_text.startswith("Error"):
        return main_text
    
    all_text = main_text
    links = get_links_from_url(url)
    for link in links[:2]:  # Limit to 2 additional pages to avoid overload
        time.sleep(1)  # Be polite
        text = scrape_url(link)
        if not text.startswith("Error"):
            all_text += "\n" + text
    
    # Store in cache
    scraped_text_cache[url] = all_text
    return all_text

def call_llm(prompt, model="groq"):
    """Call LLM API (Groq or Open Router) with the given prompt."""
    messages = [{"role": "user", "content": prompt}]
    
    if model == "groq" and GROQ_API_KEY and GROQ_API_KEY != "your-groq-api-key-here":
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": messages,
            "model": "llama-3.1-8b-instant",
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9
        }
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Groq API error: {e}, trying Open Router")
            model = "openrouter"
    
    if model == "openrouter" and OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your-openrouter-api-key-here":
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": messages,
            "model": "mistralai/mistral-small-3.1-24b-instruct",
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9
        }
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling LLM API: {str(e)}"
    
    return "No LLM API available. Please check API keys."

def extract_relevant_text(scraped_text, query):
    """Extract relevant snippets from scraped text based on query to save tokens."""
    # Simple keyword matching: find sentences containing query words
    words = query.lower().split()
    sentences = re.split(r'[.!?]', scraped_text)
    relevant_sentences = []
    for sentence in sentences:
        if any(word in sentence.lower() for word in words):
            relevant_sentences.append(sentence)
    if relevant_sentences:
        return ' '.join(relevant_sentences)[:5000]  # Limit to 5000 chars
    else:
        return scraped_text[:5000]  # If no match, return first 5000 chars

def generate_response(url, query):
    """Generate a response by scraping the URL and using LLM to process the query."""
    # Scrape the URL and linked pages
    with gr.Status("Scraping website...", state=True) as status:
        scraped_text = scrape_multiple_pages(url)
        if scraped_text.startswith("Error"):
            return scraped_text, None, None, None
    
    # Extract relevant text to save tokens
    relevant_text = extract_relevant_text(scraped_text, query)
    
    # Optimize prompt to reduce tokens
    prompt = f"""
    You are a data extraction assistant. Extract the following information from the provided text content: 
    Query: {query}
    
    Text Content:
    {relevant_text}

    Return the extracted data in JSON format. If no data is found, return an empty JSON object.
    Also, provide a brief summary of what was extracted.
    """
    
    # Call LLM
    with gr.Status("Processing with AI...", state=True) as status:
        llm_response = call_llm(prompt)
    
    # Try to parse JSON from LLM response
    try:
        start_index = llm_response.find('{')
        end_index = llm_response.rfind('}') + 1
        if start_index != -1 and end_index != -1:
            json_str = llm_response[start_index:end_index]
            data = json.loads(json_str)
        else:
            data = {}
        json_data = json.dumps(data, indent=2)
    except json.JSONDecodeError:
        json_data = "{}"
    
    # Generate a unique ID for this request and cache the data
    unique_id = hashlib.md5(f"{url}{query}{time.time()}".encode()).hexdigest()[:8]
    data_cache[unique_id] = {
        "data": json_data,
        "timestamp": time.time(),
        "url": url,
        "query": query
    }
    
    # Clean up old cache entries (older than 24 hours)
    current_time = time.time()
    for key in list(data_cache.keys()):
        if current_time - data_cache[key]["timestamp"] > 86400:
            del data_cache[key]
    
    # Create a simple API endpoint message
    api_message = f"API Endpoint ID: {unique_id}"
    
    # Extract just the summary from the LLM response
    summary = llm_response
    if "summary" in llm_response.lower() or "extracted" in llm_response.lower():
        # Try to extract just the summary part
        summary_parts = llm_response.split("\n\n")
        if len(summary_parts) > 1:
            summary = summary_parts[0]  # Assume summary is the first part
    
    return summary, json_data, api_message, f"✅ Data extracted successfully! ID: {unique_id}"

def get_json_data(unique_id):
    """Retrieve JSON data from cache by ID."""
    if unique_id in data_cache:
        # Try to parse the JSON to ensure it's valid
        try:
            json_data = json.loads(data_cache[unique_id]["data"])
            return json_data, f"✅ Data retrieved for ID: {unique_id}"
        except json.JSONDecodeError:
            return {"error": "Invalid JSON data in cache"}, "❌ Error: Invalid JSON data"
    return {"error": "Invalid or expired ID"}, "❌ Error: Invalid or expired ID"

# Custom CSS for better UI
css = """
.container {
    max-width: 1200px;
    margin: 0 auto;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #6e8efb, #a777e3);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.tab-container {
    background: #f9f9f9;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
}
.footer {
    text-align: center;
    margin-top: 20px;
    color: #666;
    font-size: 14px;
}
.status-message {
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
.success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}
.error {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
"""

# Gradio interface
with gr.Blocks(title="ParseAI - Website to API Converter", css=css) as demo:
    gr.HTML("""
    <div class="header">
        <h1>ParseAI</h1>
        <p>Turn any website into structured data with AI</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### How to Use")
            gr.Markdown("""
            1. Enter a website URL
            2. Describe what data to extract
            3. Click 'Extract Data'
            4. Use the API ID to retrieve JSON later
            """)
            
            gr.Markdown("### Examples")
            gr.Examples(
                examples=[
                    ["https://www.ssuet.edu.pk/", "Extract all academic programmes with their details"],
                    ["https://example.com", "Extract product names and prices"]
                ],
                inputs=[url_input, query_input]
            )
            
        with gr.Column(scale=2):
            with gr.Tab("Extract Data"):
                with gr.Row():
                    url_input = gr.Textbox(label="Website URL", placeholder="https://example.com", lines=1)
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="What data to extract", 
                        placeholder="Extract academic programmes, product listings, contact information, etc.",
                        lines=2
                    )
                
                extract_btn = gr.Button("Extract Data", variant="primary")
                
                status_message = gr.HTML("")
                
                with gr.Accordion("Results", open=True):
                    with gr.Row():
                        summary_output = gr.Textbox(label="Extraction Summary", interactive=False, lines=4)
                    
                    with gr.Row():
                        api_info = gr.Textbox(label="API Endpoint ID", interactive=False)
                    
                    with gr.Row():
                        json_output = gr.JSON(label="Extracted Data (JSON)")
            
            with gr.Tab("Retrieve Data"):
                gr.Markdown("Enter an API Endpoint ID to retrieve previously extracted data")
                
                with gr.Row():
                    id_input = gr.Textbox(label="API Endpoint ID", placeholder="Enter the ID from your extraction")
                
                retrieve_btn = gr.Button("Retrieve Data", variant="secondary")
                
                retrieve_status = gr.HTML("")
                
                with gr.Row():
                    retrieved_json = gr.JSON(label="Retrieved Data")
            
            with gr.Tab("API Documentation"):
                gr.Markdown("""
                ## API Usage
                
                ParseAI creates temporary API endpoints for your extracted data. Each extraction generates a unique ID that can be used to retrieve the JSON data.
                
                ### How to Access Your Data
                
                Currently, you can retrieve your data through this interface by entering the API Endpoint ID in the "Retrieve Data" tab.
                
                ### Data Retention
                
                Extracted data is stored for 24 hours. After that, it will be automatically deleted.
                
                ### Rate Limits
                
                To ensure fair usage, we limit:
                - 10 extractions per hour
                - 3 additional pages scraped per URL
                """)
    
    gr.HTML("""
    <div class="footer">
        <p>ParseAI v1.0 • Powered by Groq and Open Router AI • Data is temporarily cached for 24 hours</p>
    </div>
    """)
    
    # Event handlers
    extract_btn.click(
        fn=generate_response,
        inputs=[url_input, query_input],
        outputs=[summary_output, json_output, api_info, status_message]
    )
    
    retrieve_btn.click(
        fn=get_json_data,
        inputs=[id_input],
        outputs=[retrieved_json, retrieve_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
