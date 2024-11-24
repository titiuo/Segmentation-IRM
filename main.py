from library import *
import multiprocessing

Dcm_patients = [str("00" + str(i)) if len(str(i)) == 1 else str("0"+str(i)) for i in range(1, 21)]
Hcm_patients = [str("0"+str(i)) for i in range(21, 41)]
Minf_patients = [str("0"+str(i)) for i in range(41, 61)]
Nor_patients = [str("0"+str(i)) for i in range(61, 81)]
Rv_patients = [str("0"+str(i)) for i in range(81, 100)]
Rv_patients.append("100")
def run(start,r=5):
    for k in range(start, start + r):
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
    """ irm = Irm('001')
    step_1(irm, filtered=True, show=False)
    print(metrics(irm, show=False, write=False)) """
    for id in Dcm_patients:
        irm=Irm(id)
        step_1(irm, filtered=True, show=False)
        metrics(irm, show=False, write=True)
        print("\n\n ---------------------------------------- \n\n")
    for id in Hcm_patients:
        irm=Irm(id)
        step_1(irm, filtered=True, show=False)
        metrics(irm, show=False, write=True)
        print("\n\n ---------------------------------------- \n\n")
    for id in Minf_patients:
        irm=Irm(id)
        step_1(irm, filtered=True, show=False)
        metrics(irm, show=False, write=True)
        print("\n\n ---------------------------------------- \n\n")
    for id in Nor_patients:
        irm=Irm(id)
        step_1(irm, filtered=True, show=False)
        metrics(irm, show=False, write=True)
        print("\n\n ---------------------------------------- \n\n")
    for id in Rv_patients:
        irm=Irm(id)
        step_1(irm, filtered=True, show=False)
        metrics(irm, show=False, write=True)
        print("\n\n ---------------------------------------- \n\n)")
    for id in Rv_patients:
        irm = Irm(id)
        step_1(irm, filtered=True, show=False)
        metrics(irm, show=False, write=True)
        print("\n\n ---------------------------------------- \n\n")

"""         try:
            step_1(irm, filtered=True, show=False)
            print(metrics(irm, show=False, write=True))
        except ValueError as e:
            metrics(irm, e, show=False, write=True) """

""" processes = []
for i in range(4):
    p = multiprocessing.Process(target=run, args=(i*5+80 + 1,))
    processes.append(p)
    p.start()

for p in processes:
    p.join() """ 
    



        
