from library import *

Dcm_patients = [str("00" + str(i)) if len(str(i)) == 1 else str("0"+str(i)) for i in range(1, 21)]
Hcm_patients = [str("0"+str(i)) for i in range(21, 41)]
Minf_patients = [str("0"+str(i)) for i in range(41, 61)]
Nor_patients = [str("0"+str(i)) for i in range(61, 81)]
Rv_patients = [str("0"+str(i)) for i in range(81, 100)]

irm = Irm("001")
img = (irm.gt1[0,:,:,1] > 2).astype(np.uint8) * 255
def get_rayon(img):
    # Trouver le contour de la forme
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Supposons qu'il n'y ait qu'une seule forme
    cnt = contours[0]

    # Trouver les moments pour calculer le centre
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])  # Coordonnée x du centre
        cy = int(M["m01"] / M["m00"])  # Coordonnée y du centre
    else:
        raise ValueError("La forme est trop petite ou invalide.")

    # Calculer les distances entre le centre et chaque point du contour
    distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for [[x, y]] in cnt]

    # Estimer le rayon comme la moyenne des distances (ou médiane pour plus de robustesse)
    rayon = np.mean(distances)  # ou np.median(distances)
    pixels_2 = img.shape[0] * img.shape[1] 
    rayon = rayon**2 / pixels_2
    print(f"Rayon estimé : {rayon} pixels")
    return rayon



def get_all(irm):
    values = {}
    data = irm.gt1[0,:,:,:].astype(np.uint8)*255
    N = data.shape[-1]
    for k in range(N):
        img = data[:,:,k]
        try:
            rayon = get_rayon(img)
        except:
            rayon = 0
        percent = k/N
        values[percent]=rayon
    return values

dico = {}

for id in Dcm_patients:
    print(f"Patient {id}")
    irm=Irm(id)
    dico[str(id)]=get_all(irm)
for id in Hcm_patients:
    print(f"Patient {id}")
    irm=Irm(id)
    dico[str(id)]=get_all(irm)
for id in Minf_patients:
    print(f"Patient {id}")
    irm=Irm(id)
    dico[str(id)]=get_all(irm)
for id in Nor_patients:
    print(f"Patient {id}")
    irm=Irm(id)
    dico[str(id)]=get_all(irm)
for id in Rv_patients:
    print(f"Patient {id}")
    irm=Irm(id)  
    dico[str(id)]=get_all(irm)

with open("rayons.json", "w") as f:
    json.dump(dico, f)