import time

class Timer:
    """
    Timer class to measure time taken for a block of code

    label: label for the timer
    """
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        print(f"{self.label} time taken: {end_time - self.start_time:.2f} seconds")