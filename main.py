from library import *

if __name__ == '__main__':
    irm = Irm('002')
    step_1(irm, filtered=True, show=True)
    print(metrics(irm,show=False,write=False)) 
    """ for k in range(90,101):
    id = str(k)
    if len(id) == 1:
        id = '00'+id
    elif len(id) == 2:
        id = '0'+id
    else:
        pass
    irm = Irm(id)
    print("\n\n ---------------------------------------- \n\n")
    print(f"Processing image {id}...\n\n") 

    try:
        step_1(irm, filtered=True, show=False)
        metrics(irm,show=False)
    except ValueError as e:
        metrics(irm, e,show=False)  """

