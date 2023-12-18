# Title : Hough Transforms
# Authors:
#   Davyn Hartono - dbharton
#   Atharva Pore - apore
#   Sravya Vujjini - svujjin
#   Sanjana Jairam - sjairam


import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def hough_transform(image):

    # Detect edges using FIND_EDGES from ImageFilter library
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges)
    
    # Defining the Hough space
    height, width = edges_array.shape
    dist_max = np.ceil(np.sqrt(height**2 + width**2))
    rho_step = 1
    theta_step = np.pi/180
    accumulator = np.zeros((int(2 * dist_max // rho_step), int(180 // theta_step)))
    
    # Looping over all edge points and possible angles in range of 80 to 180 degrees
    edge_points = edges.load()
    for x in range(width):
        for y in range(height):
            if edge_points[x, y] == 255:

                 
                for theta_index in range(80, 180, 24):
                    theta = (theta_index - 90) * theta_step
                    rho = x * np.cos(theta) + y * np.sin(theta)
                    rho_index = int(round(rho / rho_step)) + int(dist_max // rho_step)
                    accumulator[rho_index, theta_index] += 1
                    
    return accumulator, dist_max, theta_step, rho_step
                

def detect_hough_lines(image, num_lines=5):
    accumulator, dist_max, theta_step, rho_step = hough_transform(image)
    
    # Locate the peaks in the Hough space
    peaks = []
    for i in range(num_lines):
        rho_index, theta_index = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        rho = (rho_index - int(dist_max // rho_step)) * rho_step
        theta = (theta_index - 90) * theta_step
        peaks.append((rho, theta))
        
        # To prevent overlapping lines, zeroed out the adjacent cells in the accumulator
        for j in range(-10, 10):
            for k in range(-10, 10):
                if theta_index+k >= 0 and theta_index+k < accumulator.shape[1] and rho_index+j >= 0 and rho_index+j < accumulator.shape[0]:
                    accumulator[rho_index+j, theta_index+k] = 0
                    
    # Drawing the detected lines on the original image
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    for rho, theta in peaks:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        draw.line((x1, y1, x2, y2), fill="red", width=4)
        
        image.save("detected_staff.png")
        
if __name__ == '__main__':
    
    if(len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                        "python3 staff_finder.py sample-input.png")
    
    image = Image.open(sys.argv[1])
    gray_image = image.convert('L')
    detect_hough_lines(gray_image)

                                                                                                                                                                                                                                                                                                                                                                                                                                                    
