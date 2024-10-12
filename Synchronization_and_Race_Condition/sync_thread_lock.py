import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(1000):
        with lock: # Acquire the lock
            counter += 1
            
if __name__ == '__main__':
    t1 = threading.Thread(target=increment)
    t2 = threading.Thread(target=increment)
    t3 = threading.Thread(target=increment)
    
    t1.start()
    t2.start()
    t3.start()
    
    t1.join()
    t2.join()
    t3.join()
    
    print(f"Counter: {counter}")