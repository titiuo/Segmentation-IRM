from library import *

patients = ["002","017","022","036","046","055","061","078","094","086"]


def get_dice_dilated(irm,e=None,N=1):
    id = irm.patient_id
    mean = 0
    predictions = []
    irm.images_processed = np.array(irm.images_processed)
    if e is not None:
        if str(e)[:26] == "Plusieurs cercles detectes":
            tmp_data = str(e)
        elif str(e) == "Too many iterations. Threshold is too high.":
            tmp_data = "Too many iterations. Threshold is too high."
        elif str(e) == "Coordinates are out of bounds.":
            tmp_data = "Coordinates are out of bounds."
        else:
            raise ValueError("Error not recognized.")
    else:
        tmp_data={}
        for k in range(irm.data.shape[-1]):
            img = binary(irm.images_processed[k,:,:])
            mask = [[0,1,0],[1,1,1],[0,1,0]]
            img = img.astype(np.uint8)
            for k in range(N):
                img = cv2.dilate(img, np.array(mask, dtype=np.uint8))
            predictions.append(img)
        gt1 = irm.gt1
        for k in range(gt1.shape[-1]):
            dice = dice_coefficient(predictions[k],gt1[0,:,:,k],False)
            #print(f"Dice coefficient for slice {k}: {dice}")
            tmp_data[f"Dice coefficient for slice {k}"] = f"{dice}"
            mean += dice
        mean /= gt1.shape[-1]
        tmp_data[f"Mean dice coefficient:"] = f"{mean}"
    return mean

all = []

for patient in patients:
    results = []
    irm = Irm(patient)
    step_1(irm, filtered=True, show=False)
    for k in range(1,6):
        value = get_dice_dilated(irm,N=k)
        results.append(value)
    all.append(results)

plt.scatter([1,2,3,4,5],all[0],color='blue',label='002')
plt.scatter([1,2,3,4,5],all[1],color='red',label='017')
plt.scatter([1,2,3,4,5],all[2],color='green',label='022')
plt.scatter([1,2,3,4,5],all[3],color='yellow',label='036')
plt.scatter([1,2,3,4,5],all[4],color='black',label='046')
plt.scatter([1,2,3,4,5],all[5],color='purple',label='055')
plt.scatter([1,2,3,4,5],all[6],color='pink',label='061')
plt.scatter([1,2,3,4,5],all[7],color='brown',label='078')
plt.scatter([1,2,3,4,5],all[8],color='orange',label='086')
plt.scatter([1,2,3,4,5],all[9],color='grey',label='094')
plt.legend()
plt.show()

    