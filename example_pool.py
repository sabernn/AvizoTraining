

from multiprocessing import Pool

def f(x):
    return x*x

def start_processes():
    with Pool() as p:
        print(p)
        print(p.map(f, [1, 2, 3]))

if __name__ == '__main__':
    start_processes()
