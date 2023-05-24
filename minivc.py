# %%
import streamlit as st
import spacy
import concurrent.futures
from functools import lru_cache
import os
import re
import json
import warnings
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse, urljoin
import openai
import PyPDF2
import requests
import tldextract
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pptx import Presentation
from tenacity import retry, stop_after_attempt, wait_random_exponential
# from tkinter import Tk, filedialog
import readppt
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

warnings.filterwarnings("ignore")
MODEL = "gpt-4"
CHUNK_SIZE=7250
ALLOWED_TLDS = {"com","app", "org", "net", "edu", "gov"}
EXCLUDED_KEYWORDS = {
    "login",
    "signup",
    "results",
    "search",
    "register",
    "account",
    "privacy",
    "terms",
    "policy",
    "disclaimer",
    "jobs",
    "careers",
    "blog"
    "contact",
    "cookie",
    "support",
    "forum",
    "cdn",
    "newsletter",
    "status",
}
FILENAME = "analyzed_data.json"

def get_links(soup, base_url):
    links = set()

    parsed_base_url = urlparse(base_url)
    ext = tldextract.extract(parsed_base_url.netloc)
    base_domain = f"{ext.domain}.{ext.suffix}"
    allowed_domains = {base_domain}

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if not href.startswith("http"):
            href = urljoin(base_url, href)
        
        parsed_url = urlparse(href)
        ext = tldextract.extract(parsed_url.netloc)
        domain = f"{ext.domain}.{ext.suffix}"

        if (base_url in href
            and domain in allowed_domains
            and ext.suffix in ALLOWED_TLDS
            and not any(keyword in href.lower() for keyword in EXCLUDED_KEYWORDS)
        ):
            links.add(href)
    print(links)
    return links

def fetch_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {url}\n{str(e)}")
        return None
    return soup

def clean_text(text):
    cleaned_text = " ".join(text.split())
    cleaned_text = re.sub(r'http\S+', '', cleaned_text)
    cleaned_text = re.sub(r'<script.*?>.*?</script>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'<style.*?>.*?</style>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = " ".join(cleaned_text.split())
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,!?/:;()%$@&\s]', '', cleaned_text)
    cleaned_text = re.sub(r'(?i)(terms\s*and\s*conditions|privacy\s*policy|copyright|blog|legal|careers|cdn*).{0,10}', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

def split_text(text, chunk_size=CHUNK_SIZE):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        tokens = nlp(sentence)
        sentence_length = len(tokens)

        if current_chunk_size + sentence_length > chunk_size:
            # Create a new chunk if adding the sentence would exceed the chunk size
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_chunk_size = sentence_length
        else:
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_chunk_size += sentence_length

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

@retry(wait=wait_random_exponential(min=2, max=20), stop=stop_after_attempt(3), reraise=True)
def base_gptcall(prompt):
    messages = [{"role": "system", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.1
    )
    return response.choices[0]['message']['content'].strip()

@retry(wait=wait_random_exponential(min=2, max=20), stop=stop_after_attempt(3), reraise=True)
def call_gpt(prompt):
    answers = []
    if len(prompt)>CHUNK_SIZE:
        textchunks = split_text(prompt)
        for chunk in textchunks:
            answer = []
            # print(len(chunk))
            # print(chunk)
            answer = base_gptcall(chunk)
            answers.append(answer)
        return ' '.join(answers)
    else:
        return base_gptcall(prompt)

def recursive_analyze(text):
    categories = [
        "Team",
        "Customers",
        "Product",
        "Market",
        "Business Model",
        "Risks",
        "Traction"
    ]
    category_explanation = [
        """The team section should include the names of the CEO, co-founders and other team members and background if available.
        Example: The CEO is Patrick Collison (ex-CEO of Auctomatic) and CTO is John Collison (ex-CTO of Auctomatic) who lead Stripe. Among its advisors include Patrick McKenzie.""",
        """The customers section should concentrate on target customer segments, industries, specific companies, and notable partnerships, without repeating information about product features or benefits.
        Example: Stripe serves businesses of all sizes across various industries, from startups like Instacart to tech giants like Amazon, providing seamless payment solutions.""",
        """The product section should describe the main product(s) or service(s), highlighting key features, benefits, use cases, and unique selling points, without discussing market size or competition.
        Example: Stripe offers a suite of payment processing services, including Stripe Payments for online transactions, Stripe Billing for subscription management, and Stripe Connect for marketplace platforms.""",
        """The market section should assess the market size, growth potential, and any adjacent opportunities, without reiterating information about the product, customers, or competition.
        Example: Stripe operates in the global digital payments market, valued at over $4 trillion, with significant growth opportunities as e-commerce and digital transactions continue to rise.""",
        """The business model section should explain the startup's revenue generation methods and pricing strategies, without focusing on product features or competition.
        Example: Stripe employs a pay-as-you-go pricing model, charging a percentage of each transaction, and offers additional features through tiered pricing plans and custom enterprise solutions.""",
        """The risks section should identify potential internal and external risks that could impact the investment, avoiding repetition of product features, benefits, or market size.
        Example: Stripe faces competition from companies like PayPal and Square, potential regulatory changes affecting the fintech industry, and evolving cyber threats and security concerns.""",
        """The traction section should cover growth rates, user engagement metrics, milestones, and future goals, without discussing the product, competition, or market size.
        Example: Stripe has experienced rapid growth, with millions of businesses using its platform, raising over $1.6 billion in funding, and expanding its services to over 40 countries."""
    ]

    category_explanation_map = dict(zip(categories, category_explanation))
    text_chunks = clean_text(text)
    text_chunks = split_text(text)
    print("The total length of all text chunks is: ")
    print(len(text_chunks))
    # Use ThreadPoolExecutor to parallelize GPT calls
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for chunk in text_chunks:
            futures.append(executor.submit(call_gpt, f"Extract all insights, names and facts from the following text as would be useful for an investment memo:\n\n{chunk}"))
        insights_lists = [future.result() for future in futures]
    insights_data = defaultdict(list)
    combined_insights = "\n".join(insights_lists)
    for category in categories:
        explanation = category_explanation_map[category]
        prompt = f"Imagining you to be writing a VC investment memo, from the following text please extract information regarding the category '{category}'. An example is here: {explanation}. If no useful information is present, please reply with 'info not available':\n\n{combined_insights}"
        summary = call_gpt(prompt)
        insights_data[category].append(summary)
    return insights_data

def link(url):
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.hostname:
        print("Invalid URL. Please provide a valid URL with a scheme (e.g., http:// or https://).")
        return None
    base_url = parsed_url.scheme + "://" + parsed_url.hostname
    soup = fetch_html(url)
    if not soup:
        return None
    links = get_links(soup, base_url)
    all_text = []
    for link in links:
        sub_soup = fetch_html(link)
        if sub_soup:
            text = clean_text(sub_soup.get_text())
            all_text.append(text)
    full_text = " ".join(all_text)
    full_text = clean_text(full_text)
    # print(f"Full text is: {full_text}")
    analyzed_data = recursive_analyze(full_text)
    return analyzed_data

def read_pdf(file):
    file.seek(0)  # move the file cursor to the beginning
    pdf_reader = PyPDF2.PdfReader(file)
    if len(pdf_reader.pages) == 0:
        raise ValueError("PDF file is empty")
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    cleaned_text = clean_text(text)
    return cleaned_text

# def get_file_input(input_type):
#     root = Tk()
#     root.withdraw()

#     if input_type == "pdf":
#         file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
#     elif input_type == "pptx":
#         file_path = filedialog.askopenfilename(filetypes=[("PowerPoint files", "*.pptx")])
#     else:
#         raise ValueError("Invalid input type")

#     return file_path

def analyze_input(input_type, company, file):
    text = ""
    if input_type == "url":
        data = link(url)
    elif input_type in ["pdf", "pptx"]:
        if not file:
            print("No file selected.")
            return

        if input_type == "pdf":
            text = read_pdf(file)
        elif input_type == "pptx":
            text = readppt.read_ppt(file.read())
        data = recursive_analyze(text)
    else:
        raise ValueError("Invalid input type")
    
    company_data = []
    for category, summary in data.items():
        edited_summary = call_gpt(f"Please rewrite this summary:{summary}")
        print(f"{category}:\n{edited_summary}\n")
        data_to_save = {
            "category": category,
            "edited_summary": edited_summary
        }
        company_data.append(data_to_save)    
    save_data(FILENAME, company, company_data)
        
    return data

def save_data(filename, company, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump({}, f)
    with open(filename, "r") as f:
        all_data = json.load(f)
    if company not in all_data:
        all_data[company] = {}
    all_data[company][timestamp] = data
    with open(filename, "w") as f:
        json.dump(all_data, f, indent=4)

def run(input_type, company, url=None, file=None):
    if not company:
        st.write("No company name provided.")
        return

    if input_type not in ["url", "pdf", "pptx"]:
        st.write(f"Invalid input type: {input_type}")
        return

    if input_type == "url":
        if not url:
            st.write("No URL provided.")
            return
        data = link(url)
    elif input_type in ["pdf", "pptx"]:
        if file is None:
            st.write("No file selected.")
            return
        data = analyze_input(input_type, company, file)
    else:
        st.write(f"Invalid input type: {input_type}")
        return

    if data is not None:
        company_data = []
        for category, summary in data.items():
            edited_summary = call_gpt(f"Please rewrite this summary:{summary}")
            st.write(f"{category}:\n{edited_summary}\n")
            data_to_save = {
                "category": category,
                "edited_summary": edited_summary
            }
            company_data.append(data_to_save)
        save_data(FILENAME, company, company_data)

def main():
    st.title("Analysis Application")
    input_type = st.selectbox("Select input type", ("url", "pdf", "pptx"))
    company = st.text_input("Enter company name")
    url = None
    file = None
    if input_type == "url":
        url = st.text_input("Enter a URL")
    elif input_type in ["pdf", "pptx"]:
        file = st.file_uploader("Please upload a file", type=[input_type])

    if st.button("Run"):
        run(input_type, company, url, file)

if __name__ == "__main__":
    main()