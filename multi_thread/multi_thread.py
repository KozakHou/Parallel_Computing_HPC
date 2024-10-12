import threading

def print_message(message):
    print(f"Thread ID: {threading.current_thread().name} - message: {message}")
    
    
if __name__ == "__main__":
    thread1 = threading.Thread(target=print_message, args=("From thread 1",))
    thread2 = threading.Thread(target=print_message, args=("From thread 2",))
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()