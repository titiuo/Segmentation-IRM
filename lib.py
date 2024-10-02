##IMPORTS
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import cv2

##FONCTIONS
##Plot
def plot_image_at_time_and_layer(patient_id, time_index, layer_index):
    if int(patient_id) < 101:
        image = tio.ScalarImage(f"../database/training/patient{patient_id}/patient{patient_id}_4d.nii.gz")
    elif 100 <int(patient_id) < 151:
        image = tio.ScalarImage(f"../database/testing/patient{patient_id}/patient{patient_id}_4d.nii.gz")
    else:
        print("Patient ID must be between 001 and 150.")
        return
    
    data = image.data.numpy()  # Convertir les données en un tableau numpy

    # Vérifier que les indices sont dans les limites
    if (0 <= time_index < data.shape[0]) and (0 <= layer_index < data.shape[-1]):
        # Extraire l'image spécifique pour le temps et la couche donnés
        image_to_plot = data[time_index, :, :, layer_index]

        # Afficher l'image
        plt.imshow(image_to_plot, cmap='gray')  # Utiliser la carte de couleurs appropriée
        plt.title(f'Image at Time {time_index}, Layer {layer_index}')
        plt.axis('off')  # Cacher les axes
        plt.show()
    else:
        print("Les indices fournis sont hors limites.")

def plot_all_layers_at_time(patient_id, time_index):
    if int(patient_id) < 101:
        image = tio.ScalarImage(f"../database/training/patient{patient_id}/patient{patient_id}_4d.nii.gz")
    elif 100 < int(patient_id) < 151:
        image = tio.ScalarImage(f"../database/testing/patient{patient_id}/patient{patient_id}_4d.nii.gz")
    else:
        print("Patient ID must be between 001 and 150.")
        return

    data = image.data.numpy()  # Convertir les données en un tableau numpy

    # Nombre de couches
    num_layers = data.shape[-1]

    # Calculer les dimensions de la grille (par exemple 3x3, 4x4 selon le nombre de couches)
    cols = int(np.ceil(np.sqrt(num_layers)))  # Nombre de colonnes (racine carrée du nombre de couches)
    rows = int(np.ceil(num_layers / cols))  # Nombre de lignes

    # Créer une figure avec plusieurs sous-graphes en grille
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for layer_index in range(num_layers):
        # Extraire l'image spécifique pour le temps et la couche donnés
        image_to_plot = data[time_index, :, :, layer_index]
        
        # Déterminer la position dans la grille
        row = layer_index // cols
        col = layer_index % cols
        
        # Afficher l'image dans la sous-figure correspondante
        axes[row, col].imshow(image_to_plot, cmap='gray')
        axes[row, col].set_title(f'Layer {layer_index}')
        axes[row, col].axis('off')  # Cacher les axes

    # Supprimer les axes vides s'il y a moins d'images que de sous-graphes
    for i in range(layer_index + 1, rows * cols):
        fig.delaxes(axes.flatten()[i])

    # Ajuster la disposition des sous-graphes
    plt.tight_layout()
    plt.show()

def plot_all_times_for_layer(patient_id, layer_index):
    if int(patient_id) < 101:
        image = tio.ScalarImage(f"../database/training/patient{patient_id}/patient{patient_id}_4d.nii.gz")
    elif 100 < int(patient_id) < 151:
        image = tio.ScalarImage(f"../database/testing/patient{patient_id}/patient{patient_id}_4d.nii.gz")
    else:
        print("Patient ID must be between 001 and 150.")
        return

    data = image.data.numpy()  # Convertir les données en un tableau numpy

    # Nombre d'instants (frames)
    num_times = data.shape[0]

    # Calculer les dimensions de la grille
    cols = int(np.ceil(np.sqrt(num_times)))  # Nombre de colonnes
    rows = int(np.ceil(num_times / cols))  # Nombre de lignes

    # Créer une figure avec plusieurs sous-graphes en grille
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for time_index in range(num_times):
        # Extraire l'image spécifique pour le temps et la couche donnés
        image_to_plot = data[time_index, :, :, layer_index]
        
        # Déterminer la position dans la grille
        row = time_index // cols
        col = time_index % cols

        # Afficher l'image dans la sous-figure correspondante
        axes[row, col].imshow(image_to_plot, cmap='gray')
        axes[row, col].set_title(f'Time {time_index}')
        axes[row, col].axis('off')  # Cacher les axes

    # Supprimer les axes vides s'il y a moins d'images que de sous-graphes
    for i in range(time_index + 1, rows * cols):
        fig.delaxes(axes.flatten()[i])

    # Ajuster la disposition des sous-graphes
    plt.tight_layout()
    plt.show()

def patient_info(patient_id):
    if int(patient_id) < 101:
        filename = f"../database/training/patient{patient_id}/Info.cfg"
    elif 100 <int(patient_id) < 151:
        image = f"../database/testing/patient{patient_id}/Info.cfg"
    else:
        print("Patient ID must be between 001 and 150.")
        return
    
    info_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            if key=="ED" or key=="ES":
                value = int(value)
            info_dict[key] = value
            #print(f'{key}: {value}')

    return info_dict

def hough_transform(image,show=True):
    """
    Applique la transformation de Hough sur une image et renvoie les coordonnées des centres des cercles détectés.
    
    Args:
        image (numpy.ndarray): Image sur laquelle appliquer la transformation de Hough (format numpy).
        
    Returns:
        image_with_circles (numpy.ndarray): Image avec les cercles dessinés.
        seed_points (list of tuples): Liste des coordonnées (x, y) des centres des cercles détectés.
    """
    # Convertir l'image en 8 bits pour OpenCV
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Appliquer le flou pour réduire le bruit
    blurred_image = cv2.medianBlur(image_8bit, 5)

    # Détecter les cercles via la transformation de Hough
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=10, maxRadius=100)

    seed_points = []  # Liste pour stocker les centres des cercles (seed points)

    # Si des cercles sont détectés, les dessiner et stocker les centres
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            seed_points.append((x, y))  # Ajouter les coordonnées du centre
            cv2.circle(blurred_image, (x, y), r, (255, 0, 0), 2)  # Dessiner le cercle
            cv2.circle(blurred_image, (x, y), 2, (0, 255, 0), 3)  # Dessiner le centre

    # Afficher l'image avec les cercles
    if show:
        plt.imshow(blurred_image, cmap='gray')
        plt.title("Hough Transform on Image with Detected Circles")
        plt.axis('off')
        plt.show()
    # Retourner l'image avec les cercles et les seed points
    return blurred_image, seed_points

def absolute_difference_Ed_ES(patient_id,slice_index):
    if int(patient_id) < 101:
        image = tio.ScalarImage(f"../database/training/patient{patient_id}/patient{patient_id}_4d.nii.gz")
    elif 100 < int(patient_id) < 151:
        image = tio.ScalarImage(f"../database/testing/patient{patient_id}/patient{patient_id}_4d.nii.gz")
    else:
        print("Patient ID must be between 001 et 150.")
        return
    # Calculer la différence en valeurs absolues
    data = image.data.numpy()
    image1 = data[patient_info(patient_id)["ED"], :, :, slice_index]
    image2 = data[patient_info(patient_id)["ES"], :, :, slice_index]
    abs_diff = np.abs(image1 - image2)
    
    return abs_diff

def set_pixel_red(image_gray, x, y):
    """
    Met le pixel (x, y) en rouge dans une image en niveaux de gris.

    Args:
        image_gray (numpy.ndarray): Image en niveaux de gris (2D array).
        x (int): Coordonnée x du pixel à mettre en rouge.
        y (int): Coordonnée y du pixel à mettre en rouge.

    Returns:
        numpy.ndarray: Image avec le pixel spécifié en rouge.
    """
    # Convertir l'image en niveaux de gris en image RGB
    image_rgb = np.stack((image_gray,)*3, axis=-1)  # Créer un tableau RGB en dupliquant l'image en niveaux de gris

    # Mettre le pixel spécifié en rouge
    image_rgb[y, x] = [255, 0, 0]  # [R, G, B]
    image_rgb[y,x+1]=[255,0,0]
    image_rgb[y+1,x]=[255,0,0]
    image_rgb[y,x-1]=[255,0,0]
    image_rgb[y-1,x]=[255,0,0]
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def step_1(patient_id):

    def energy(p,sigma,mean,p_CoG,intensity,w=11):
        return np.sqrt((2*sigma/(w-1)*np.sqrt((p[0]-p_CoG[0])**2+p[0]-p_CoG[0])**2)**2 + (intensity-mean)**2)

    if int(patient_id) < 101:
        image = tio.ScalarImage(f"../database/training/patient{patient_id}/patient{patient_id}_4d.nii.gz")
    elif 100 < int(patient_id) < 151:
        image = tio.ScalarImage(f"../database/testing/patient{patient_id}/patient{patient_id}_4d.nii.gz")
    else:
        print("Patient ID must be between 001 et 150.")
        return
    data=image.data.numpy()
    middle_slice_index = image.shape[-1] // 2
    abs_diff=absolute_difference_Ed_ES(patient_id, middle_slice_index)
    image_with_circles, seed_points = hough_transform(abs_diff, show=False)
    if len(seed_points) > 1:
        print("Plusieurs cercles détécté.")
        return
    t_ED=patient_info(patient_id)["ED"]
    t_ES=patient_info(patient_id)["ES"]
    seed_points={(t_ED,middle_slice_index):seed_points[0],(t_ES,middle_slice_index):seed_points[0]}
    w=11
    center_x,center_y=seed_points[(t_ED,middle_slice_index)]
    set_pixel_red(data[t_ED,:,:,middle_slice_index], center_x, center_y)
    for slice in range(middle_slice_index+1,data.shape[-1]):
        Energies={}
        for dy in range(-w//2, w//2):
            for dx in range(-w//2, w//2):
                # Coordonnées du pixel dans la fenêtre
                pixel_x = center_x + dx
                pixel_y = center_y + dy
                Energies[(pixel_x,pixel_y)]=energy((pixel_x,pixel_y),sigma=1,mean=0,p_CoG=(center_x,center_y),intensity=data[t_ED,pixel_x,pixel_y,slice],w=w)
        min_energy_pixel = min(Energies, key=Energies.get)
        seed_points[(t_ED,slice)]=min_energy_pixel

        set_pixel_red(data[t_ED,:,:,slice], min_energy_pixel[0], min_energy_pixel[1])
        


# Exemple d'utilisation
#patient_info("001")
#plot_image_at_time_and_layer("001", 0,10)
#plot_image_at_time_and_layer("001",12,0)
#plot_all_layers_at_time("001", 0)
#plot_all_times_for_layer("001", 5)

# Afficher l'image de différence absolue entre ED et ES pour un patient et une couche donnés
#plt.imshow(absolute_difference_Ed_ES("001", 5), cmap='gray')
#plt.title("Absolute Difference Image")
#plt.axis('off')
#plt.show()

#print(hough_transform(absolute_difference_Ed_ES("001", 5))[1])

step_1("001")