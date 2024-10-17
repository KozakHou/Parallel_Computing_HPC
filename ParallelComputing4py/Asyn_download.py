import asyncio
import aiohttp

urls = [
    'https://www.example.com',
    'https://www.python.org',
    'https://www.openai.com',
    # Add more URLs here
]

async def download(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f'{url} downloaded {response.content_length} bytes')
            return await response.text()
        
async def main():
    tasks = [download(url) for url in urls]
    await asyncio.gather(*tasks)
    
if __name__ == "__main__":
    asyncio.run(main())