import numpy as np
import cv2
import math


#generate Gaussian Kernel
def gaussianKernel(sigma):
    
    size = math.ceil(3*sigma)
    
    h, k = np.mgrid[-size:size+1, -size:size+1]

    g = (np.exp(-(h*h + k*k)/(2*sigma*sigma)))/(2*np.pi*sigma*sigma)
    
    return g

#convolution 2D
def convolution2d(img, kernel):

    #grad is the output image
    grad = np.array(img)
    size = kernel.shape[0]//2
    
    for i in range(img.shape[0]):
        
        for j in range(img.shape[1]):

            #record for one pixel
            gradP = 0
            for h in range(-size,size+1):
                for k in range(-size,size+1):

                    try:
                        gradP += kernel[h, k] * img[i-h, j-k]
                        
                    except:
                        
                        gradP += kernel[h, k] * img[i, j]

            grad[i,j] = gradP

    return grad

#combine the gaussian kernel and the img with convolution
def gaussianBlur(img, sigma):
    
    A = gaussianKernel(sigma)
    return convolution2d(img, A)

            
#generate sobel kernel and use sobel filter to produce gradient magnitude and gradient orientation
def sobel(img, threshold):

    imageX = np.empty(shape=(img.shape[0],img.shape[1]))
    imageY = np.empty(shape=(img.shape[0],img.shape[1]))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try:
                imageX[i, j] = (int(img[i-1, j-1]) - int(img[i-1, j+1])) + (2*(int(img[i, j-1])-int(img[i,j+1]))) + (int(img[i+1, j-1])- int(img[i+1, j+1]))

            except:
                imageX[i, j] = 0
            try:
                imageY[i, j] = (int(img[i-1, j-1]) - int(img[i+1, j-1])) + (2*(int(img[i-1, j])-int(img[i+1,j]))) + (int(img[i-1, j+1])- int(img[i+1, j+1]))

            except:
                imageY[i, j] = 0

    #combine sobel x image and sobel y image
    output = cv2.convertScaleAbs((np.sqrt(np.square(np.float32(imageX)) + np.square(np.float32(imageY)))))
    
    #make image lighter
    output = output/output.max() * 255

    #gradient orientation by arc tan
    theta = np.arctan2(np.float32(imageY), np.float32(imageX))
    
    #threshold
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i, j] < threshold:
                output[i, j] = 0
    return output, theta

#main progress function
def myEdgeFilter(img, sigma, threshold):
    h = img.shape[0]
    w = img.shape[1]
    ## smooth by Gaussian
    blur = gaussianBlur(img,sigma)

    img, theta = sobel(blur, threshold)
    
    edgePixel = np.zeros((h,w), dtype=np.int32)

    #convert angle to degree(up to 180)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    #follow the direction of edge to use non-maximum suppression
    for i in range(1,h-1):
        for j in range(1,w-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    edgePixel[i,j] = img[i,j]
                else:
                    edgePixel[i,j] = 0

            except:
                 edgePixel[i,j] = 0
    
    return edgePixel


#main function
img = cv2.imread('img0.jpg',0)

#sigma = 1 #thresold range 0-255
edge = np.uint8(myEdgeFilter(img,1, 100))

cv2.imshow("sobel",edge)

cv2.imwrite("edge_detection_result.png", edge)

cv2.waitKey(0) # waits until a key is pressed

cv2.destroyAllWindows() # destroys the window showing image

