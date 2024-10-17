import multiprocessing


def calculate_square(numbers, result, index):
    total = 0
    for n in numbers:
        total += n*n
    result[index] = total


if __name__ == "__main__":
    numbers = list(range(1, 100001))
    manager = multiprocessing.Manager()
    result = manager.dict()
    
    p1 = multiprocessing.Process(target=calculate_square, args=(numbers[:50000], result, 0))
    p2 = multiprocessing.Process(target=calculate_square, args=(numbers[50000:], result, 1))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    print(f"Sum of squares: {result[0] + result[1]}")
