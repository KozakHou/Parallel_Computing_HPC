import threading
import requests


uel = urls = [
    'https://www.example.com',
    'https://www.python.org',
    'https://www.openai.com',
    # Add more URLs here
]


def download(url):
    response = requests.get(url)
    print(f'{url} downloaded {len(response.content)} bytes')


threads = []
for url in urls:
    thread = threading.Thread(target=download, args = (url, ))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()