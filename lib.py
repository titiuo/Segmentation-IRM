from library import *
import cv2

irm = Irm('008')

working_set = irm.abs_diff

working_set = working_set.astype('uint8')
edges = cv2.Canny(working_set, 100, 200)
#working_set = edges

hough_transform = hough(edges)

plt.figure(2)
plt.imshow(working_set, cmap='gray')
plt.figure(3)
plt.scatter(hough_transform[1], hough_transform[0])
plt.imshow(edges, cmap='gray')
plt.show()

