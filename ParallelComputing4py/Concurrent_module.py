from concurrent.futures import ThreadPoolExecutor
import requests


urls = [
    'https://www.example.com',
    'https://www.python.org',
    'https://www.openai.com',
    # Add more URLs here
]

def download(url):
    response = requests.get(url)
    print(f'{url} downloaded {len(response.content)} bytes')
    
with ThreadPoolExecutor(max_workers = 5) as executor:
    executor.map(download, urls)
    
## ThreadPoolExecutor can automatically manage the threads to be created and destroyed.