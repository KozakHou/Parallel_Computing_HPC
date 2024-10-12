import multiprocessing

def sum_numbers(start, end, queue):
    summation = 0
    for i in range(start, end + 1):
        summation += i 
        
    queue.put(summation)
    print(f"Process ID: {multiprocessing.current_process().name} - Sum from {start} to {end} is {summation}")
    
    
if __name__ == "__main__":
    # create queue 
    queue = multiprocessing.Queue()
    # create processes list 
    processes = []
    ranges = [(1, 250), (251, 500), (501, 750), (751, 1000)]
    
    for idx, (start, end) in enumerate(ranges):
        process = multiprocessing.Process(target=sum_numbers, args=(start, end, queue), name=f"Process-{idx + 1}")
        processes.append(process)
        process.start()
        
    # wait for all processes to finish
    for process in processes:
        process.join()
        
    total_sum = 0
    while not queue.empty():
        total_sum += queue.get()
        
    print(f"Total sum from 1 to 1000 is {total_sum}")
    
## This is real prarllelisim
## since every process has its own memory space on independent CPU cores which bypasses the GIL limitation


## GIL: Global Interpreter Lock
## GIL ensures that only one thread executes Python bytecode at a time
## which means that even in a multi-core systems, Python threads are not executed in for CPU bound tasks
