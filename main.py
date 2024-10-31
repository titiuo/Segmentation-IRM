from library import *
import multiprocessing

def run(start):
    for k in range(start, start + 10):
        id = str(k)
        if len(id) == 1:
            id = '00' + id
        elif len(id) == 2:
            id = '0' + id
        else:
            pass
        irm = Irm(id)
        print("\n\n ---------------------------------------- \n\n")
        print(f"Processing image {id}...\n\n")

        try:
            step_1(irm, filtered=True, show=False)
            metrics(irm, show=False, write=True)
        except ValueError as e:
            metrics(irm, e, show=False, write=True)


if __name__ == '__main__':
    irm = Irm('001')
    step_1(irm, filtered=True, show=True)
    """ processes = []
    for i in range(10):
        p = multiprocessing.Process(target=run, args=(i * 10 + 1,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()  """
    



        
