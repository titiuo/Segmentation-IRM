from library import *
import cv2

irm = Irm('035')
irm.show_slices(0)
first_slice = irm.data[0,:,:,0]
std = np.std(first_slice)
mean = np.mean(first_slice)
print('Mean: ', mean)
print('Std: ', std)
if std < 10:
    lowThresh = 15*std
elif 10 <= std < 26:
    lowThresh = 20*std
else:
    lowThresh = 15*std
high_thresh = 1.5*lowThresh
edges = cv2.Canny(first_slice.astype('uint8'), 180, 300)
u,v = hough(edges)
plt.imshow(edges, cmap='gray')
plt.scatter(v,u, color='red')
plt.show()


working_set = irm.abs_diff.astype('uint8')
high_thresh, thresh_im = cv2.threshold(working_set, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lowThresh = 0.5*high_thresh
std = np.std(working_set)
mean = np.mean(working_set)
print('Mean: ', mean)
print('Std: ', std)
if std < 10:
    lowThresh = 15*std
elif 10 <= std < 26:
    lowThresh = 20*std
else:
    lowThresh = 15*std
high_thresh = 1.5*lowThresh

edges = cv2.Canny(working_set, lowThresh/2, high_thresh/2)
#working_set = edges


y,x = hough(edges)

w = (x+v)/2
z = (y+u)/2



plt.figure(1)
plt.imshow(working_set, cmap='gray')
plt.scatter(x,y, color='red')


plt.figure(2)
plt.scatter(x, y, color='red')
plt.scatter(v,u, color='blue')
plt.scatter(w,z, color='green')
plt.imshow(edges, cmap='gray')
plt.show()

"""
Note of observations:

when std close to 20, lowthresh = 20*std, highthresh = 1.5*lowthresh
when close to 30, lowthresh = 15*std, highthresh = 1.5*lowthresh

issue with 038: taking 10*std and 20*std works fine
"""