from library import *
import cv2

irm = Irm('003')

working_set = irm.abs_diff.astype('uint8')
high_thresh, thresh_im = cv2.threshold(working_set, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lowThresh = 0.5*high_thresh
print(high_thresh, lowThresh)
edges = cv2.Canny(working_set, lowThresh, high_thresh)
#working_set = edges

hough_transform = hough(edges)

plt.figure(2)
plt.imshow(working_set, cmap='gray')
plt.figure(3)
plt.scatter(hough_transform[1], hough_transform[0])
plt.imshow(edges, cmap='gray')
plt.show()

