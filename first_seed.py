from library import *
import multiprocessing



Dcm_patients = [str("00" + str(i)) if len(str(i)) == 1 else str("0"+str(i)) for i in range(1, 21)]
Hcm_patients = [str("0"+str(i)) for i in range(21, 41)]
Minf_patients = [str("0"+str(i)) for i in range(41, 61)]
Nor_patients = [str("0"+str(i)) for i in range(61, 81)]
Rv_patients = [str("0"+str(i)) for i in range(81, 100)]
Rv_patients.append("100")

def main(irm,show='False'):

    print(f"\n\n------------------- Patient ID: {irm.patient_id} -------------------\n\n")
    
    if int(irm.patient_id) < 101:
        pass
    elif 100 < int(irm.patient_id) < 151:
        pass
    else:
        print("Patient ID must be between 001 et 150.")
        return
    
    middle_slice_index = irm.middle_slice

    values = new_hough(irm,show=True)
    x = np.median([val[0] for val in values])
    y = np.median([val[1] for val in values])
    initial_seed_point = [(int(x),int(y))]
    if len(initial_seed_point) > 1:
        raise ValueError(f"Plusieurs cercles detectes : {len(initial_seed_point)}")
        

    t_ED = irm.t_ED
    t_ES = irm.t_ES
    irm.seed_points={(t_ED,middle_slice_index):initial_seed_point[0],(t_ES,middle_slice_index):initial_seed_point[0]}
    center_y,center_x=irm.seed_points[(t_ED,middle_slice_index)]


    expected_image = irm.gt1[0,:,:,middle_slice_index]

    if expected_image[initial_seed_point[0][0],initial_seed_point[0][1]] != 3:
        if show=='All':
            plt.figure()
            for val in values:
                plt.scatter(val[1],val[0],color='blue')
                plt.scatter(center_x,center_y,color='red')
            plt.imshow(expected_image,cmap='gray')
            plt.title(f"Seed of {irm.patient_id} point in the LV.")
        if show=='True':
            plt.scatter(center_x,center_y,color='red')
            plt.imshow(expected_image,cmap='gray')
            plt.title(f"Seed of {irm.patient_id} point in the LV.")
            plt.show()
        print(f"Seed point of {irm.patient_id} not in the LV.")
        return False
    else: 
        print(f"Seed point of {irm.patient_id} in the LV.")
        if show=='All':
            plt.figure()
            for val in values:
                plt.scatter(val[1],val[0],color='blue')
                plt.scatter(center_x,center_y,color='red')
                plt.scatter(center_x,center_y,color='red')
            plt.imshow(expected_image,cmap='gray')  
            plt.title(f"Seed of {irm.patient_id} point in the LV.")
        if show=='True':
            plt.scatter(center_x,center_y,color='red')
            plt.imshow(expected_image,cmap='gray')
            plt.title(f"Seed of {irm.patient_id} point in the LV.")
            plt.show()
     
        return True
    
if __name__ == '__main__':
    """ irm = Irm("052")
    main(irm,show='True')  """
    count = 0
    for id in Dcm_patients:
        irm = Irm(id)
        flag = main(irm,show=False)
        if not flag:
            count += 1
            if count > 10:
                break
    for id in Hcm_patients:
        irm = Irm(id)
        flag = main(irm,show=False)
        if not flag:
            count += 1 
            if count > 10:
                break
    for id in Minf_patients:
        irm = Irm(id)
        flag = main(irm,show=False)
        if not flag:
            count += 1
            if count > 10:
                break
    for id in Nor_patients:
        irm = Irm(id)
        flag = main(irm,show=False)
        if not flag:
            count += 1
            if count > 10:
                break
    for id in Rv_patients:
        irm = Irm(id)
        flag = main(irm,show=False)
        if not flag:
            count += 1
            if count > 10:
                break 
    print(f"{count} images not in rv.") 
 
    









