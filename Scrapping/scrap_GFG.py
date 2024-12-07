import requests
from bs4 import BeautifulSoup
import os
import json
from datetime import datetime

def download_html():
    # URL of the page
    url = 'https://www.geeksforgeeks.org/variables-in-c/?ref=lbp'
    
    try:
        # Adding headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Send GET request to the URL
        print(f"Attempting to download HTML from: {url}")
        response = requests.get(url, headers=headers)
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        print("Successfully downloaded HTML content.")
        return response.text  # Return HTML content as a string
    
    except requests.RequestException as e:
        print(f"Error downloading the webpage: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def extract_qa_code(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Define lists to store extracted data
    qa_data = []
    
    # List of keywords to filter out irrelevant sections
    filter_keywords = [
        'similar reads', 'more reads', 'recommended',
        'related tutorials', 'related articles', 'further reading',
        'references', 'see also', 'additional resources'
    ]
    
    # Find all headings that represent questions (h2, h3) and their corresponding answers
    headings = soup.find_all(['h2', 'h3'])
    
    for heading in headings:
        # Convert heading text to lowercase for case-insensitive filtering
        heading_text = heading.get_text(strip=True).lower()
        
        # Skip irrelevant sections
        if any(keyword in heading_text for keyword in filter_keywords):
            continue
        
        question = heading.get_text(strip=True)
        answer = ""
        code_snippet = ""
        
        # Find the next siblings (paragraphs or blockquotes) as the answer
        next_tag = heading.find_next_sibling()
        
        while next_tag and next_tag.name in ['p', 'blockquote', 'div']:
            answer += next_tag.get_text(strip=True) + "\n"
            next_tag = next_tag.find_next_sibling()
        
        # Find the code block related to the answer (if exists)
        code_block = heading.find_next('pre')
        if code_block:
            code_snippet = code_block.get_text(strip=True)
        
        # Only add non-empty entries
        if question.strip() and (answer.strip() or code_snippet.strip()):
            qa_data.append({
                'question': question,
                'answer': answer.strip(),
                'code': code_snippet.strip() if code_snippet else None
            })
    
    return qa_data

#Extraction happens on the go
if __name__ == "__main__":
    print("Starting HTML download and extraction process...")
    
    #Download HTML
    html_content = download_html()
    
    if html_content:
        #Extract Q&A and code snippets
        extracted_data = extract_qa_code(html_content)
        
        #Save extracted data to JSON file
        output_filename = f'tutor_bot_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(f'Scrapping/Scrapped_data/${output_filename}', 'w', encoding='utf-8') as json_file:
            json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)
        
        print(f"\nData extraction completed! Saved as {output_filename}.")
        print(f"File size: {os.path.getsize(f'Scrapping/Scrapped_data/${output_filename}') / 1024:.2f} KB")
    else:
        print("\nDownload failed. Please check the error messages above.")