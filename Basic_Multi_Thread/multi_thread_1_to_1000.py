import threading

# Define a global list to store partial sums
partial_sums = []

# Define a lock object to protect the shared resource
lock = threading.Lock()

def sum_numbers(start, end):
    summation = 0
    for i in range(start, end + 1):
        summation += i 

    # Save the partial sum to the global list with lock
    with lock:
        partial_sums.append(summation)
        print(f"Thread ID: {threading.current_thread().name} - Sum from {start} to {end} is {summation}")    
        

if __name__ =="__main__":
    threads = []
    ranges = [(1, 250), (251, 500), (501, 750), (751, 1000)]
    
    for idx, (start, end) in enumerate(ranges):
        thread = threading.Thread(target=sum_numbers, args = (start, end), name = f"Thread-{idx + 1}")
        threads.append(thread)
        thread.start()
    
    # wait for all threads to finish
    for thread in threads:
        thread.join()
        
    total_sum = sum(partial_sums)
    print(f"Total sum from 1 to 1000 is {total_sum}")