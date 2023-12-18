# Title : Optical Music Recognition
# Authors:
#   Davyn Hartono - dbharton
#   Atharva Pore - apore
#   Sravya Vujjini - svujjin
#   Sanjana Jairam - sjairam

# Reference: https://github.com/imdeep2905/Notch-Filter-for-Image-Processing

from PIL import Image
import sys
from scipy import fftpack
import numpy as np

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                        "python3 remove_noise.py noisy_pichu.png")


    def notch_filter(shape, size, dx, dy, teta):
        """
        Creating filter that consists of 2 mirrored square dots.

        Description
        ----------------
        This function produces matrix with the same size of image.
        The value of matrix is one except the dots which has value less than 1.

        :param shape: image size
        :param size: filter size
        :param dx: horizontal distance a dot to the center of image
        :param dy: vertical distance a dot to the center of image
        :param teta: value for the filter (should be less than 1)
        :return: matrix
        """
        width, height = shape
        wc = width / 2
        hc = height / 2
        filter = np.ones(shape)

        # Padding for adjusting filter size
        pad = (size - 1) / 2

        # Dot 1
        filter[int(np.round(hc+dy-pad,0)):int(np.round(hc+dy+pad,0)),int(np.round(wc+dx-pad,0)):int(np.round(wc+dx+pad,0))] = teta

        # Dot 2
        filter[int(np.round(hc-dy-pad,0)):int(np.round(hc-dy+pad)),int(np.round(wc-dx-pad,0)):int(np.round(wc-dx+pad,0))] = teta

        return filter

    img = Image.open(sys.argv[1]).convert('L')

    # Fourier transform of input image
    fft_input = fftpack.fftshift(fftpack.fft2(img))

    # Filter
    filter = notch_filter(img.size,24,17,17,teta=0.1)

    # Perform matrix multiplication in spectrum space
    fft_new = fft_input * filter

    # Visualizing the result on spectrum space
    fft_new_img = Image.fromarray((np.log(abs(fft_new)) * 255 / np.amax(np.log(abs(fft_new)))).astype(np.uint8))
    fft_new_img.save('fft_denoise_pichu.png')

    # Transform the matrix into spatial space
    new_img = abs(fftpack.ifft2(fftpack.ifftshift(fft_new)))
    new_img = Image.fromarray(new_img.astype(np.uint8))
    new_img.save('denoise_pichu.png')

