####################        IMPORTS       ####################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
import torchio as tio
from collections import deque
import json



####################        CLASS DEFINITION       ####################

class Irm():
    def __init__(self,patient_id):
        self.patient_id = patient_id
        self.image = tio.ScalarImage(f"../database/training/patient{patient_id}/patient{patient_id}_4d.nii.gz")
        self.data = self.image.data.numpy()
        self.info = patient_info(patient_id)
        self.seed_points = {}
        self.midde_slice = self.data.shape[-1]//2
        self.t_ED=patient_info(self.patient_id)["ED"]-1
        self.t_ES=patient_info(self.patient_id)["ES"]-1
        self.images_processed = []
        self.initial_seed_point = None
        self.set_of_slices = {} #key: time index, value: array of slices
        self.hough_transform_image = None
        for k in range(self.data.shape[0]):
            self.set_of_slices[k] = np.array([self.data[k,:,:,i] for i in range(self.data.shape[3])])
        self.set_of_times = {} #key: layer index, value: array of times
        for k in range(self.data.shape[3]):
            self.set_of_times[k] = np.array([self.data[i,:,:,k] for i in range(self.data.shape[0])])
        self.abs_diff = absolute_difference_Ed_ES(patient_id, self.midde_slice)
        self.gt1 = tio.ScalarImage(f"../database/training/patient{patient_id}/patient{patient_id}_frame01_gt.nii.gz").data.numpy()
        #self.gt2 = tio.ScalarImage(f"../database/training/patient{patient_id}/patient{patient_id}_frame12_gt.nii.gz").data.numpy()
        self.mean_dice = None

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
            self.hough_transform_image = image_rgb
        # Retourner l'image avec les cercles et les seed points
        return blurred_image, seed_points
    
    def dice_coefficient(self):
        first = self.images_processed[self.t_ED]
        second = self.images_processed[self.t_ES]


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

    def save_image(self, event):
        """Save the current image to a file."""
        filename = f'image_{self.current_index}.png'
        plt.imsave(filename, self.data[self.current_index, :, :], cmap='gray')
        print(f'Saved image {filename}')

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
    #image_rgb = np.stack((image_gray,)*3, axis=-1)  # Créer un tableau RGB en dupliquant l'image en niveaux de gris
    if image_gray.dtype != np.uint8:
        image_gray = cv2.convertScaleAbs(image_gray)  # Convertir en 8-bit unsigned

    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)  # Convertir en RGB tout en conservant les niveaux de gris

    # Mettre le pixel spécifié en rouge
    image_rgb[y, x] = [255, 0, 0]  # [R, G, B]
    image_rgb[y,x+1]=[255,0,0]
    image_rgb[y+1,x]=[255,0,0]
    image_rgb[y,x-1]=[255,0,0]
    image_rgb[y-1,x]=[255,0,0]
    if show:
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()  
    return image_rgb

def step_1(irm,show=False,filtered=False):
    def energy(p,sigma,mean,p_CoG,intensity,w=11):
        return np.sqrt((2*sigma/(w-1)*np.sqrt((p[0]-p_CoG[0])**2+p[0]-p_CoG[0])**2)**2 + (intensity-mean)**2)

    if int(irm.patient_id) < 101:
        pass
    elif 100 < int(irm.patient_id) < 151:
        pass
    else:
        print("Patient ID must be between 001 et 150.")
        return
    
    def get_next_seed():
        Energies={}
        for dy in range(-w//2, w//2):
            for dx in range(-w//2, w//2):
                # Coordonnées du pixel dans la fenêtre
                pixel_x = center_x + dx
                pixel_y = center_y + dy
                Energies[(pixel_x,pixel_y)]=energy((pixel_x,pixel_y),std,mean,p_CoG=(center_x,center_y),intensity=data[current_time,pixel_x,pixel_y,current_slice],w=w)
        min_energy_pixel = min(Energies, key=Energies.get)
        print(f"Minimum energy pixel: {min_energy_pixel} at slice {current_slice}.")
        return min_energy_pixel
    
    
    temporary_dictionnary = {} #used to store slicez in the right order
    data = irm.data
    middle_slice_index = irm.midde_slice
    
    image_32f = irm.abs_diff.astype(np.float32)
    working_set = (cv2.bilateralFilter(image_32f, 2, 75, 200) > 30).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)

    # Application de la fermeture (dilatation suivie d'une érosion)
    closed_image = cv2.morphologyEx(working_set, cv2.MORPH_CLOSE, kernel)
    initial_seed_point = [hough(closed_image,w=30)]
    if len(initial_seed_point) > 1:
        raise ValueError(f"Plusieurs cercles detectes : {len(initial_seed_point)}")
        

    t_ED = irm.t_ED
    t_ES = irm.t_ES
    irm.seed_points={(t_ED,middle_slice_index):initial_seed_point[0],(t_ES,middle_slice_index):initial_seed_point[0]}
    w=11
    center_y,center_x=irm.seed_points[(t_ED,middle_slice_index)]
    

    #irm.images_processed.append(tmp)
    to_process=[(t_ED,middle_slice_index),(t_ED,middle_slice_index+1),(t_ED,middle_slice_index-1)]
    while to_process:
        current_time, current_slice = to_process.pop(0)
        print(f"Processing time {current_time}, slice {current_slice}.")
        image_segmented,region = region_growing_adaptive(irm,current_time,center_y,center_x,current_slice,filtered=filtered, nb_neighbours=4)

        image_segmented[center_y,center_x] = [0,0,255]
        image_segmented[center_y,center_x+1] = [0,0,255]
        image_segmented[center_y+1,center_x] = [0,0,255]
        image_segmented[center_y,center_x-1] = [0,0,255]
        image_segmented[center_y-1,center_x] = [0,0,255]

        temporary_dictionnary[(current_time,current_slice)] = image_segmented
    
        if (current_time, current_slice) in irm.seed_points:
            mean = np.mean([data[current_time,x,y,current_slice] for x,y in region])
            std=np.std([data[current_time,x,y,current_slice] for x,y in region])
            #print(mean)
            continue

        #min_energy_pixel = get_next_seed()
        min_energy_pixel = barycentre(irm,current_time, current_slice, region)
        print(f"New seed point: {min_energy_pixel} for slice: {current_slice}.")
        irm.seed_points[(current_time,current_slice)]=min_energy_pixel
        center_x,center_y=min_energy_pixel
        image_segmented[center_y,center_x] = [255,0,0]
        #tmp = set_pixel_red(data[current_time,:,:,current_slice], min_energy_pixel[0], min_energy_pixel[1])
        
        #irm.images_processed.append(tmp)
        if current_slice+1<data.shape[-1] and (current_time,current_slice+1) not in irm.seed_points:
            to_process.append((current_time,current_slice+1))
        if current_slice-1>=0 and (current_time,current_slice-1) not in irm.seed_points:
            to_process.append((current_time,current_slice-1))
        l = [data[current_time,x,y,current_slice] for x,y in region]
        #print(l)
        mean = np.mean(l)
        #print(mean)
    for k in range(irm.data.shape[-1]):
        irm.images_processed.append(temporary_dictionnary[(t_ED,k)])
    if show:
        irm.images_processed = np.array(irm.images_processed)
        bulk_plot(irm.images_processed)
    return 

def bulk_plot(data):
    nav = ImageNavigator(data)
    return nav

def ker_gau(s):
    ss=int(max(3,2*np.round(2.5*s)+1))
    ms=(ss-1)//2
    X=np.arange(-ms,ms+0.99)
    y=np.exp(-X**2/2/s**2)
    out=y.reshape((ss,1))@y.reshape((1,ss))
    out=out/out.sum()
    return out


def filtre_lineaire(im,mask):
    """ renvoie la convolution de l'image avec le mask. Le calcul se fait en 
utilisant la transformee de Fourier et est donc circulaire.  Fonctionne seulement pour 
les images en niveau de gris.
"""
    fft2=np.fft.fft2
    ifft2=np.fft.ifft2
    (y,x)=im.shape
    (ym,xm)=mask.shape
    mm=np.zeros((y,x))
    mm[:ym,:xm]=mask
    fout=(fft2(im)*fft2(mm))
    # on fait une translation pour ne pas avoir de decalage de l'image
    # pour un mask de taille impair ce sera parfait, sinon, il y a toujours un decalage de 1/2
    mm[:ym,:xm]=0
    y2=int(np.round(ym/2-0.5))
    x2=int(np.round(xm/2-0.5))
    mm[y2,x2]=1
    out=np.real(ifft2(fout*np.conj(fft2(mm))))
    return out



def region_growing(irm, t,x ,y ,z, threshold=20, filtered=False):
    s = 0.56
    if filtered:
        working_set = filtre_lineaire(irm.data[t,:,:,z],ker_gau(s))
    else:
        working_set = irm.data[t,:,:,z]
    initial_x,initial_y = x,y
    to_explore = [(x,y)]
    explored = []
    edge= []
    region=[(x,y)]
    image_processed = irm.data[t,:,:,z]
    while to_explore:
        if len(explored)>2500:
            raise ValueError("Too many iterations. Threshold is too high.")
        x,y = to_explore.pop(0)
        if not -1< x < irm.data.shape[1] or not -1< y < irm.data.shape[2]:
            raise ValueError("Coordinates are out of bounds.")
        for couple in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if couple not in explored and couple not in edge:
                if abs(working_set[x,y] -working_set[couple[0],couple[1]]) < threshold:
                    to_explore.append(couple)
                    region.append(couple)
                else:
                    edge.append(couple)
            else:
                continue 
            explored.append(couple)
    image_rgb = np.stack((image_processed,)*3, axis=-1)
    for couple in edge:
        image_rgb[couple[0],couple[1]] = [255,0,0]
    print(f'Number of pixels in the region: {len(region)} for s = {s}')
    return image_rgb,region

def region_growing_adaptive(irm, t,x ,y ,z, threshold=15, filtered=False, nb_neighbours=8):
    s = 0.56
    """ if filtered:
        working_set = filtre_lineaire(irm.data[t,:,:,z],ker_gau(s))
    else:
        working_set = irm.data[t,:,:,z] """
    # Convert the image to 32-bit float format
    if filtered:
        image_32f = irm.data[t,:,:,z].astype(np.float32)
        working_set = cv2.bilateralFilter(image_32f, 9, 75, 75)
    else:
        working_set = irm.data[t,:,:,z]
    print(f"shape of working_set : {working_set.shape}")
    initial_x,initial_y = x,y
    to_explore = [(x,y)]
    explored = []
    edge= []
    region=[(x,y)]
    image_processed = irm.data[t,:,:,z]
    while to_explore:
        if len(explored)>2500:
            to_explore = [(initial_x,initial_y)]
            explored = []
            edge= []
            region=[(initial_x,initial_y)]
            #print("Too many iterations. Threshold is too high.")
            threshold-=1
        x,y = to_explore.pop(0)
        if not -1< x < irm.data.shape[1] or not -1< y < irm.data.shape[2]:
            raise ValueError("Coordinates are out of bounds.")
        if nb_neighbours == 4:
            neighbours = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        elif nb_neighbours == 8:
            neighbours = [(x+1,y),(x-1,y),(x,y+1),(x,y-1), (x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)]
        for couple in neighbours:
            if couple not in explored and couple not in edge and -1<couple[0]<irm.data.shape[1] and -1<couple[1]<irm.data.shape[2]:
                if abs(working_set[x,y] -working_set[couple[0],couple[1]]) < threshold:
                    to_explore.append(couple)
                    region.append(couple)
                else:
                    edge.append(couple)
            else:
                continue 
            explored.append(couple)
    region = dilate(region)
    image_rgb = np.stack((image_processed,)*3, axis=-1)
    """ for couple in edge:
        image_rgb[couple[0],couple[1]] = [255,0,0]  """
    for couple in region:
        try:
            image_rgb[couple[0],couple[1]] = [0,128,0]
        except:
            pass
    #print(f'Number of pixels in the region: {len(region)} for s = {s}')
    return image_rgb,region

def barycentre(irm, t, z, region):
    A = np.zeros((irm.data.shape[1], irm.data.shape[2]))
    
    if not region:
        raise ValueError("Region is empty, cannot compute barycentre.")
    
    for couple in region:
        x, y = couple
        A[x, y] = 1

    total_sum = np.sum(A)
    if total_sum == 0:
        raise ValueError("Sum of the binary image is zero, cannot compute barycentre.")

    x = np.sum(A * np.arange(A.shape[0]).reshape(-1, 1)) / total_sum
    y = np.sum(A * np.arange(A.shape[1]).reshape(1, -1)) / total_sum

    """ plt.scatter([y], [x], color='red')  # Visualiser le barycentre
    plt.imshow(A)
    plt.title(f"Barycentre for slice {z}")
    plt.show() """

    #print(f"Barycentre: ({y}, {x})")
    return (int(y), int(x))


def binary(image):
    im = np.zeros((image.shape[0],image.shape[1]))
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if (image[x,y] == [0,128,0]).all() or (image[x,y] == [0,0,255]).all():
                im[x,y] = 1
            else:
                pass
    return im


def dice_coefficient(image1,image2,show=False):
    image2 = image2 > 2.
    intersection = np.logical_and(image1,image2)
    diff = np.abs(image1-intersection)
    if show:
        fig, axs = plt.subplots(1,4 , figsize=(15, 5))
        axs[0].imshow(image1, cmap='gray')
        axs[0].set_title('Image 1')
        axs[0].axis('off')

        axs[1].imshow(image2, cmap='gray')
        axs[1].set_title('Image 2')
        axs[1].axis('off')

        axs[2].imshow(intersection, cmap='gray')
        axs[2].set_title('Intersection')
        axs[2].axis('off')

        axs[3].imshow(diff, cmap='gray')
        axs[3].set_title('Difference between our segmentation and real segmentation')
        axs[3].axis('off')
        plt.show()
    return 2. * intersection.sum() / (image1.sum() + image2.sum())


def metrics(irm, e=None,show = False,write=True):
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
            predictions.append(binary(irm.images_processed[k,:,:]))
        gt1 = irm.gt1
        for k in range(gt1.shape[-1]):
            dice = dice_coefficient(predictions[k],gt1[0,:,:,k],show)
            tmp_data[f"Dice coefficient for slice {k}"] = f"{dice}"
            mean += dice
        mean /= gt1.shape[-1]
        tmp_data[f"Mean dice coefficient:"] = f"{mean}"
    if write:
        try:
            with open('logs.json', "r") as file:
                data = json.load(file)
        except FileNotFoundError:
        # Si le fichier n'existe pas, initialiser un tableau vide
            data = {}

        # Ajouter les nouvelles données
        data[str(id)] = tmp_data

        # Réécrire le fichier JSON avec les nouvelles données
        with open('logs.json', "w") as file:
            json.dump(data, file, indent=4)
    return mean
    
    

def dilate(region):
    tmp = region.copy()
    for pixel in tmp:
        x,y = pixel
        for couple in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if couple not in tmp:
                region.append(couple)
    return region
   

irm=Irm("100")
image_32f = irm.abs_diff.astype(np.float32)
working_set = (cv2.bilateralFilter(image_32f, 2, 75, 200) > 30).astype(np.uint8)
kernel = np.ones((5,5), np.uint8)

# Application de la fermeture (dilatation suivie d'une érosion)
closed_image = cv2.morphologyEx(working_set, cv2.MORPH_CLOSE, kernel)

def get_window(image,w=75):
    region = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            region.append((x,y))
    total_intensity = sum([image[couple[0], couple[1]] for couple in region])
    x = sum([couple[0] * image[couple[0], couple[1]] for couple in region]) / total_intensity
    y = sum([couple[1] * image[couple[0], couple[1]] for couple in region]) / total_intensity
    x = int(x)
    y = int(y)
    windowed = image[x-w:x+w,y-w:y+w]
    return windowed,(x,y)



radius=15
thickness=1
def hough(closed_image,w=75,thickness=1):
    closed_image,(x_bary,y_bary) = get_window(closed_image,w)
    grad_x = cv2.Sobel(closed_image.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(closed_image.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=5)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    closed_image = grad_magnitude >35
    shape = closed_image.shape
    ima_intens=np.zeros((15,shape[0],shape[1]))
    for r in range(10,25):
        for a in range(closed_image.shape[0]):
            for b in range(closed_image.shape[1]):
                if closed_image[a,b]==True:
                    mask=np.zeros(closed_image.shape)
                    cv2.circle(mask,(a,b),r,255,thickness)
                    ima_intens[r-10][mask==255]+=1 
    """ plt.imshow(ima_intens)
    plt.show()  """
    return np.unravel_index(np.argmax(ima_intens),ima_intens.shape)[1:]+np.array([x_bary-w,y_bary-w])

""" point = hough(closed_image.astype(bool),w=30)



print(point)

plt.figure(1)
plt.imshow(closed_image) 
plt.figure(2)
plt.imshow(working_set)
plt.show() """
