import threading

## use multi-threading to download multiple urls content
## and save them into a shared list safely

def download_url(url, shared_list):
    # download the content of the url
    content = f"Content of {url}"
    shared_list.append(content)
    
    
    
if __name__ == "__main__":
    urls = ["url1", "url2", "url3"]
    shared_list = []
    threads = []
    
    for url in urls:
        thread = threading.Thread(target=download_url, args=(url, shared_list))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print(shared_list)
