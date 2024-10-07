####################        IMPORTS       ####################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
import torchio as tio


####################        CLASS DEFINITION       ####################

class Irm():
    def __init__(self,patient_id):
        self.patient_id = patient_id
        self.image = tio.ScalarImage(f"../database/training/patient{patient_id}/patient{patient_id}_4d.nii.gz")
        self.data = self.image.data.numpy()
        self.info = patient_info(patient_id)
        self.seed_points = {}
        self.matrice_image = np.zeros((self.data.shape[0],self.data.shape[1],self.data.shape[2],3,self.data.shape[3]))
        self.midde_slice = self.data.shape[-1]//2
        self.images_with_circles = []
        self.initial_seed_point = None
        self.set_of_slices = {} #key: time index, value: array of slices
        for k in range(self.data.shape[0]):
            self.set_of_slices[k] = np.array([self.data[k,:,:,i] for i in range(self.data.shape[3])])
        self.set_of_times = {} #key: layer index, value: array of times
        for k in range(self.data.shape[3]):
            self.set_of_times[k] = np.array([self.data[i,:,:,k] for i in range(self.data.shape[0])])
        self.abs_diff = absolute_difference_Ed_ES(patient_id, self.midde_slice)

    def show_slices(self, time_index):
        """this function plots all layers of the 4D image at a specific time index"""
        nav = ImageNavigator(self.set_of_slices[time_index])
        return nav
    
    def show_times(self, layer_index):
        """this function plots all time indexes of the 4D image at a specific layer index"""
        nav = ImageNavigator(self.set_of_times[layer_index],type="time")
        return nav
    
    def plot_image_at_time_and_layer(self, time_index, layer_index):
        """this function plots the image of a patient at a specific time and layer"""
        if int(self.patient_id) < 101:
            image = tio.ScalarImage(f"../database/training/patient{self.patient_id}/patient{self.patient_id}_4d.nii.gz")
        elif 100 <int(self.patient_id) < 151:
            image = tio.ScalarImage(f"../database/testing/patient{self.patient_id}/patient{self.patient_id}_4d.nii.gz")
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

    def hough_transform(self,image,show=True):
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
            image_rgb = set_pixel_red(blurred_image, seed_points[0][0], seed_points[0][1],show)
            self.images_with_circles.append(image_rgb)
        # Retourner l'image avec les cercles et les seed points
        return blurred_image, seed_points


class ImageNavigator:
    def __init__(self, data,type="slice"):
        self.data = data
        self.num_slices = data.shape[0] #could also represent time
        self.current_index = 0

        # Initialize figure and axes
        self.fig, self.ax = plt.subplots()
        self.image = self.ax.imshow(self.data[self.current_index, :, :], cmap='gray')
        self.ax.set_title(f'{type}: {self.current_index}')
        self.ax.axis('off')

        # Add buttons for navigation
        axprev = plt.axes([0.7, 0.01, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.01, 0.1, 0.075])
        self.bprev = Button(axprev, 'Previous')
        self.bnext = Button(axnext, 'Next')
        self.bprev.on_clicked(lambda event: self.prev_image(event, type))
        self.bnext.on_clicked(lambda event: self.next_image(event, type))

        plt.show()

    def update_image(self,type):
        self.image.set_data(self.data[self.current_index, :, :])
        self.ax.set_title(f'{type}: {self.current_index}')
        self.fig.canvas.draw()

    def prev_image(self, event,type):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image(type)

    def next_image(self, event,type):
        if self.current_index < self.num_slices - 1:
            self.current_index += 1
            self.update_image(type)




####################        FUNCTIONS       ####################

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




def set_pixel_red(image_gray, x, y,show=False):
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
    """     if show:
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()  """
    return image_rgb

def step_1(irm,show=False):
    def energy(p,sigma,mean,p_CoG,intensity,w=11):
        return np.sqrt((2*sigma/(w-1)*np.sqrt((p[0]-p_CoG[0])**2+p[0]-p_CoG[0])**2)**2 + (intensity-mean)**2)

    if int(irm.patient_id) < 101:
        pass
    elif 100 < int(irm.patient_id) < 151:
        pass
    else:
        print("Patient ID must be between 001 et 150.")
        return
    
    

    data = irm.data
    matrice_image = np.zeros((data.shape[0],data.shape[1],data.shape[2],3,data.shape[3]))
    middle_slice_index = irm.midde_slice
    abs_diff = irm.abs_diff
    image_with_circles, initial_seed_point = irm.hough_transform(abs_diff, show)
    

    if len(initial_seed_point) > 1:
        print("Plusieurs cercles détécté.")
        return
    t_ED=patient_info(irm.patient_id)["ED"]-1
    t_ES=patient_info(irm.patient_id)["ES"]-1
    seed_points={(t_ED,middle_slice_index):initial_seed_point[0],(t_ES,middle_slice_index):initial_seed_point[0]}
    w=11
    center_x,center_y=seed_points[(t_ED,middle_slice_index)]
    matrice_image[t_ED,:,:,:,middle_slice_index]=set_pixel_red(data[t_ED,:,:,middle_slice_index], center_x, center_y,show)

    to_process=[(t_ED,middle_slice_index),(t_ED,middle_slice_index+1),(t_ED,middle_slice_index-1)]
    while to_process:
        current_time, current_slice = to_process.pop(0)
        print(f"Processing time {current_time}, slice {current_slice}.")
        #step2()

        if (current_time, current_slice) in seed_points:
            continue

        Energies={}
        for dy in range(-w//2, w//2):
            for dx in range(-w//2, w//2):
                # Coordonnées du pixel dans la fenêtre
                pixel_x = center_x + dx
                pixel_y = center_y + dy
                Energies[(pixel_x,pixel_y)]=energy((pixel_x,pixel_y),sigma=1,mean=0,p_CoG=(center_x,center_y),intensity=data[current_time,pixel_x,pixel_y,current_slice],w=w)
        min_energy_pixel = min(Energies, key=Energies.get)
        seed_points[(current_time,current_slice)]=min_energy_pixel
        center_x,center_y=min_energy_pixel

        matrice_image[current_time,:,:,:,current_slice]=set_pixel_red(data[current_time,:,:,current_slice], min_energy_pixel[0], min_energy_pixel[1],show)
        if current_slice+1<data.shape[-1] and (current_time,current_slice+1) not in seed_points:
            to_process.append((current_time,current_slice+1))
        if current_slice-1>=0 and (current_time,current_slice-1) not in seed_points:
            to_process.append((current_time,current_slice-1))
    return seed_points,matrice_image

irm = Irm("001")
step_1(irm,show=True)

irm.images_with_circles = np.array(irm.images_with_circles)
print(irm.images_with_circles.shape)
print(irm.images_with_circles)

plt.imshow(irm.images_with_circles[0,:,:,0])
plt.axis('off')
plt.show()
