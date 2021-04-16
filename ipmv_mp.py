

'''
GAURAVI SONKUSARE
IPMV MINI PROJECT - IMAGE FILTERING PROGRAM
'''

'''
STEPS:
STEP 1.  IMPORT ALL THE LIBRARIES
STEP 2.  DISPLAY THE IMAGE TO BE OPERATED ON
STEP 3.  MAKE AN INDEX OF ALL THE OPERATIONS THAT CAN BE PERFORMED IN THIS PROGRAM
STEP 4.  DEFINE FUNCTIONS FOR ALL THE OPERATIONS
STEP 5.  DISPLAY THE INDEX AND ALLOW CHOOSING OF THE OPERATION TO BE PERFORMED
STEP 6.  PERFORM THE OPERATION
STEP 7.  SHOW THE OUTPUT IMAGE
STEP 8.  REDIRECT TO THE INDEX
STEP 9.  PERFORM ANOTHER OPERATION ON A NEW IMAGE IF SELECTED
STEP 10. REDIRECT TO THE INDEX. EXIT IF CHOOSEN
'''


#START

#STEP 1.  IMPORT ALL THE LIBRARIES

import numpy as np
import matplotlib.pyplot as plt 
import cv2
from PIL import Image

#STEP 2.  DISPLAY THE IMAGE TO BE OPERATED ON

print('Welcome!!')
img = Image.open("B2DBy.jpg")
print('Your Original image:')
plt.imshow(img) 
plt.show() 

#STEP 4.  DEFINE FUNCTIONS FOR ALL THE OPERATIONS

#1.FLIP HORIZONTALLY:

def FLIP_HORIZONTALLY(img): 
    img = Image.open("B2DBy.jpg")
    horizontal_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    print('Given image:')
    plt.imshow(img) 
    plt.show() 
    print('Horizontally flipped image:')
    plt.imshow(horizontal_flip) 
    plt.show()
    new_img=horizontal_flip
    lst(new_img)

#2.FLIP VERTICALLY:

def FLIP_VERTICALLY(img):
    img = Image.open("B2DBy.jpg")
    vertical_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
    print('Given image:')
    plt.imshow(img) 
    plt.show() 
    print('Vertically flipped image:')
    plt.imshow(vertical_flip) 
    plt.show()
    new_img=vertical_flip
    lst(new_img)

#3.ROTATE 90DEGREE ANTICLOCKWISE:

def ROTATE_90DEGREE_ANTICLOCKWISE(img):
    img = Image.open("B2DBy.jpg")
    rotate_90degree_anticlockwise = img.transpose(Image.ROTATE_90)
    print('Given image:')
    plt.imshow(img) 
    plt.show() 
    print('90 degree anticlockwise rotated image:')
    plt.imshow(rotate_90degree_anticlockwise) 
    plt.show() 
    new_img=rotate_90degree_anticlockwise
    lst(new_img)
   
#4.CHANGE TO GREEN COLOUR PLANE:    

def CHANGE_TO_GREEN_COLOUR_PLANE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    row,col,plane = img.shape
    temp = np.zeros((row,col,plane),np.uint8)
    temp[:,:,1] = img[:,:,1]
    plt.imshow(temp) 
    print('Green colour plane image:')
    plt.show() 
    new_img=temp
    lst(new_img)

#5.CHANGE TO BLUE COLOUR PLANE:

def CHANGE_TO_BLUE_COLOUR_PLANE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    row,col,plane = img.shape
    temp = np.zeros((row,col,plane),np.uint8)
    temp[:,:,2] = img[:,:,2]
    plt.imshow(temp) 
    print('Blue colour plane image:')
    plt.show() 
    new_img=temp
    lst(new_img)

#6.CHANGE TO RED COLOUR PLANE:

def CHANGE_TO_RED_COLOUR_PLANE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    row,col,plane = img.shape
    temp = np.zeros((row,col,plane),np.uint8)
    temp[:,:,0] = img[:,:,0]
    plt.imshow(temp) 
    print('Red colour plane image:')
    plt.show() 
    new_img=temp
    lst(new_img)

#7.CHANGE TO GREY COLOUR PLANE:    

def CHANGE_TO_GREY_COLOUR_PLANE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    plt.imshow(gray) 
    print('Gray colour image:')
    plt.show()   
    new_img=gray
    lst(new_img)

#8.NEGATE THE IMAGE:

def NEGATE_THE_IMAGE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show()  
    height, width, _ = img.shape 
    for i in range(0, height - 1): 
        for j in range(0, width - 1):
            pixel = img[i, j] 
            pixel[0] = 255 - pixel[0] # red 
            pixel[1] = 255 - pixel[1] # green
            pixel[2] = 255 - pixel[2] # blue
            img[i, j] = pixel 
    print('Negated image:')
    plt.imshow(img) 
    plt.show()  
    new_img=img
    lst(new_img)

#9.ERODE THE IMAGE:    

def ERODE_THE_IMAGE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    kernel = np.ones((5,5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    plt.imshow(img_erosion) 
    print('Eroded image:')
    plt.show()
    new_img=img_erosion
    lst(new_img)

#10.DILATE THE IMAGE:    

def DILATE_THE_IMAGE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    plt.imshow(img_dilation) 
    print('Dialated image:')
    plt.show() 
    new_img=img_dilation
    lst(new_img)
  
#11.SKELETONIZE THE IMAGE:

def SKELETONIZE_THE_IMAGE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    kernel = np.ones((5,5),np.uint8)
    skeletonization = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    plt.imshow(skeletonization) 
    print('Skeletonization of image:')
    plt.show() 
    new_img=skeletonization
    lst(new_img)

#12.OPENING OF THE IMAGE:

def OPENING_OF_THE_IMAGEI(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening) 
    print('Opening of image:')
    plt.show() 
    new_img=opening
    lst(new_img)

#13.CLOSING OF THE IMAGE:

def CLOSING_OF_THE_IMAGE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closing) 
    print('Closing of image:')
    plt.show() 
    new_img=closing
    lst(new_img)

#14.CONTRAST STRETCHING THE IMAGE:

def CONTRAST_STRETCHING_THE_IMAGE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show()
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img = cv2.LUT(img, table)
    print('Contrast stretched image:')
    plt.imshow(img) 
    plt.show()
    new_img=img
    lst(new_img)

#15.LOG TRANSFORMATION THE IMAGE:

def LOG_TRANSFORMATION_THE_IMAGE(img):
    img = cv2.imread('B2DBy.jpg') 
    plt.imshow(img) 
    print('Given image:')
    plt.show()   
    c = 255/(np.log(1 + np.max(img)))
    log_transformed_img = c * np.log(1 + img)
    log_transformed_img = np.array(log_transformed_img, dtype = np.uint8)
    print('Log transformed image:')
    plt.imshow(log_transformed_img) 
    plt.show() 
    new_img=log_transformed_img
    lst(new_img)

#16.GAMMA CORRECT THE IMAGE:

def GAMMA_CORRECT_THE_IMAGE(img):
    img = cv2.imread('B2DBy.jpg') 
    plt.imshow(img) 
    print('Given image:')
    plt.show()   
    for gamma in [0.1, 0.5, 1.2, 2.2]:
        gamma_corrected_image = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
    print('Gamma corrected image:')
    plt.imshow(gamma_corrected_image) 
    plt.show() 
    new_img=gamma_corrected_image
    lst(new_img)
 
#17.ADD GAUSSIAN NOISE TO THE IMAGE:

def ADD_GAUSSIAN_NOISE_TO_THE_IMAGE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    img_gauss = cv2.add(img,gauss)
    plt.imshow(img_gauss) 
    print('Gaussian noise image:')
    plt.show()
    new_img=img_gauss
    lst(new_img)

#18.APPLY LOW PASS AVERAGE FILTERING:    

def APPLY_LOW_PASS_AVERAGE_FILTERING(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show()  
    blur = cv2.blur(img,(5,5))
    print('Low pass Averaging filtered (smoothened/blurred) image:')
    plt.imshow(blur) 
    plt.show() 
    new_img=blur
    lst(new_img)

#19.APPLY LOW PASS MEDIAN FILTERING(SMOOTHING):    

def APPLY_LOW_PASS_MEDIAN_FILTERING(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show()  
    median = cv2.medianBlur(img,5)
    print('Low pass Median filtered image:')
    plt.imshow(median) 
    plt.show() 
    new_img=median
    lst(new_img)
 
#20.APPLY GAUSSIAN FILTERING:

def APPLY_GAUSSIAN_FILTERING(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show()  
    gaussian = cv2.GaussianBlur(img,(5,5),0)
    print('Gaussian filtered image:')
    plt.imshow(gaussian) 
    plt.show() 
    new_img=gaussian
    lst(new_img)
    
#21.APPLY HIGH PASS FILTERING:

def APPLY_HIGH_PASS_FILTERING(img):
    img = cv2.imread('B2DBy.jpg') 
    plt.imshow(img) 
    print('Given image:')
    plt.show()  
    img1 = np.array([[0.0, -1.0, 0.0], 
                       [-1.0, 5.0, -1.0],
                       [0.0, -1.0, 0.0]])
    img1 = img1/(np.sum(img1) if np.sum(img1)!=0 else 1)
    img2 = cv2.filter2D(img,-1,img1)
    plt.imshow(img2) 
    print('High pass filtered image:')
    plt.show()
    new_img=img2
    lst(new_img)

#22.APPLY HIGH BOOST FILTERING(1SHARPENING):    

def APPLY_HIGH_BOOST_FILTERING(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show()   
    img = cv2.resize(img , (400 , 400))
    kernel = np.array([[-1 , -1 , -1] , [-1 , 9 , -1] ,[-1 , -1 , -1]])
    sharp_img = cv2.filter2D(img , -1 , kernel = kernel)
    plt.imshow(sharp_img) 
    print('High boost filtered (sharpened) image:')
    plt.show() 
    new_img=sharp_img
    lst(new_img)

#23.APPLY MEAN FILTERING:

def APPLY_MEAN_FILTERING(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    figure_size = 9
    new_image = cv2.blur(image,(figure_size, figure_size)) 
    img1=(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB))
    plt.imshow(img1)
    print('Mean filtered image:')
    plt.show()
    new_img=img1
    lst(new_img)

#24.CROPPING THE IMAGE:    

def CROPPING_THE_IMAGE(img):
    img = cv2.imread('B2DBy.jpg', 1) 
    plt.imshow(img) 
    print('Given image:')
    plt.show() 
    crop = img[50:150,20:200]
    plt.imshow(crop) 
    print('Cropped image:')
    plt.show()
    new_img=crop
    lst(new_img)

#STEP 3.  MAKE AN INDEX OF ALL THE OPERATIONS THAT CAN BE PERFORMED IN THIS PROGRAM   

def lst(img): 

#STEP 5.  DISPLAY THE INDEX AND ALLOW CHOOSING OF THE OPERATION TO BE PERFORMED
    
    print('Choose an operation to be performed on the above image from the below given Index:')
    print('1.FLIP HORIZONTALLY')
    print('2.FLIP VERTICALLY')
    print('3.ROTATE 90DEGREE ANTICLOCKWISE')
    print('4.CHANGE TO GREEN COLOUR PLANE')
    print('5.CHANGE TO BLUE COLOUR PLANE')
    print('6.CHANGE TO RED COLOUR PLANE')
    print('7.CHANGE TO GREY COLOUR PLANE')
    print('8.NEGATE THE IMAGE')
    print('9.ERODE THE IMAGE')
    print('10.DILATE THE IMAGE')
    print('11.SKELETONIZE THE IMAGE')
    print('12.OPENING OF THE IMAGE')
    print('13.CLOSING OF THE IMAGE')
    print('14.CONTRAST STRETCHING THE IMAGE')
    print('15.LOG TRANSFORMATION THE IMAGE')
    print('16.GAMMA CORRECT THE IMAGE')
    print('17.ADD GAUSSIAN NOISE TO THE IMAGE')
    print('18.APPLY LOW PASS AVERAGE FILTERING')
    print('19.APPLY LOW PASS MEDIAN FILTERING(SMOOTHING)')
    print('20.APPLY GAUSSIAN FILTERING')
    print('21.APPLY HIGH PASS FILTERING')
    print('22.APPLY HIGH BOOST FILTERING(1SHARPENING)')
    print('23.APPLY MEAN FILTERING')
    print('24.CROPPING THE IMAGE')
    print('25.EXIT EDITING')

#STEP 7.  SHOW THE OUTPUT IMAGE
#STEP 8.  REDIRECT TO THE INDEX
#STEP 9.  PERFORM ANOTHER OPERATION ON A NEW IMAGE IF SELECTED
    
    n=int(input())
    if n==1:
        FLIP_HORIZONTALLY(img)
    elif n==2:
        FLIP_VERTICALLY(img)
    elif n==3:
        ROTATE_90DEGREE_ANTICLOCKWISE(img)
    elif n==4:
        CHANGE_TO_GREEN_COLOUR_PLANE(img)
    elif n==5:
        CHANGE_TO_BLUE_COLOUR_PLANE(img)
    elif n==6:
        CHANGE_TO_RED_COLOUR_PLANE(img)
    elif n==7:
        CHANGE_TO_GREY_COLOUR_PLANE(img)
    elif n==8:
        NEGATE_THE_IMAGE(img)
    elif n==9:
        ERODE_THE_IMAGE(img)
    elif n==10:
        DILATE_THE_IMAGE(img)
    elif n==11:
        SKELETONIZE_THE_IMAGE(img)
    elif n==12:
        OPENING_OF_THE_IMAGEI(img)
    elif n==13:
        CLOSING_OF_THE_IMAGE(img)
    elif n==14:
        CONTRAST_STRETCHING_THE_IMAGE(img)
    elif n==15:
        LOG_TRANSFORMATION_THE_IMAGE(img)
    elif n==16:
        GAMMA_CORRECT_THE_IMAGE(img)
    elif n==17:
        ADD_GAUSSIAN_NOISE_TO_THE_IMAGE(img)
    elif n==18:
        APPLY_LOW_PASS_AVERAGE_FILTERING(img)
    elif n==19:
        APPLY_LOW_PASS_MEDIAN_FILTERING(img)
    elif n==20:
        APPLY_GAUSSIAN_FILTERING(img)
    elif n==21:
        APPLY_HIGH_PASS_FILTERING(img)
    elif n==22:
        APPLY_HIGH_BOOST_FILTERING(img)
    elif n==23:
        APPLY_MEAN_FILTERING(img)
    elif n==24:
        CROPPING_THE_IMAGE(img)
        
#STEP 10. REDIRECT TO THE INDEX. EXIT IF CHOOSEN 
#25.EXIT EDITING:   
        
    elif n==25:
        print('Thank you for using this program.')
    else :
        print('Please enter a value within the given index. Thank you!')
        img = Image.open("B2DBy.jpg")
        print('Original image:')
        plt.imshow(img) 
        plt.show() 
        lst(img)
        
#STEP 6.  PERFORM THE OPERATION
      
lst(img)

#END