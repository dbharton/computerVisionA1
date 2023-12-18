# Title : Optical Music Recognition
# Authors:
#   Davyn Hartono - dbharton
#   Atharva Pore - apore
#   Sravya Vujjini - svujjin
#   Sanjana Jairam - sjairam


import sys
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
from scipy import stats

# Some of the general workflows refered to https://medium.com/mlearning-ai/optical-music-recognition-6257a9bcca52
# Additional refference for image processing algorithm follows text book "Burger & Burge, Principles of Digital Image Processing - Fundamental techniques"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                     "python3 omr.py music1.png")

    # Binary image : replacing value above or equal to 128 with 255, otherwise 0
    # 128 is chosen as it is the middle value on 0-255 pixel range
    def binarize(image):
        """
        Binarizing image.

        Description
        ----------------
        Replacing the pixels value which < 127.5 as 0 (black pixels), otherwise as 255 (white pixels)
        Parameters
        ----------------
        :param image: input image
        :return: binarized image
        """
        # input size
        input_w = image.width
        input_h = image.height

        # Binarizing
        for x in range(input_w):
            for y in range(input_h):
                p = image.getpixel((x,y))
                if p >= 128:
                    p = 255
                    image.putpixel((x,y), p)
                else:
                    p = 0
                    image.putpixel((x,y), p)
        return image

    def padding(image, padding_size, mode='edge'):
        """
        padd an image using numpy.pad.

        Parameters
        ----------------
        :param image: The image to be padded.
        :param padding_size: tuple of padding size (before,after) at the beginning and at the end of the image.
        :param mode: Padding method 'constant', 'edge', 'symmetric', 'reflect', 'maximum', 'minimum', 'median', 'mean'.
        :return: new padded image.
        """
        image = np.array(image).astype(np.float64)
        image = np.pad(image, padding_size, mode=mode)
        image = Image.fromarray(np.uint8(image))

        return image

    # Finding staves line in input image
    # Algorithm ref: Burger & Burge, Principles of Digital Image Processing - Fundamental techniques
    # Assumption: staves are horizontal
    def staves(image, teta):
        """
        Detecting the staves row coordinate & applying non maximum suppression for multiple detection in a single line.

        Parameters
        ----------------
        :param image: input image.
        :param teta: threshold which determine an object to be predicted as a line.
        :return: list of the row coordinate of staves.
        """
        # Finding series of black pixel horizontally
        acc = np.zeros(image.height)
        for y in range(image.height):
            for x in range(image.width):
                p = image.getpixel((x, y))
                if p < 255:
                    acc[y] += 1

        acc = np.where(acc >= teta, acc, 0)

        # Using non maxima suppression method, if checked node is < max(neighbors) it will be suppressed
        # Add padding & check neighbors accumulator
        acc = np.pad(acc,5)
        for i in range(5,len(acc)-5):
            neighbors = np.copy(acc[i-5:i+6])
            neighbors[5] = 0
            if acc[i] < np.amax(neighbors):
                acc[i] = 0

        # Remove padding
        acc = acc[5:len(acc)-5]
        return np.where(acc > 0)[0]


    def staves_gap(staves_pos):
        """
        Calculating the gap between stave

        Description
        ----------------
        The location of staves might slightly shifted, resulting in the fact that the gap will not be constant.
        Hence, we calculate each gap than we use the mode of those calculated gaps.

        Parameters
        ----------------
        :param staves_pos: the row coordinate of staves.
        :return: integer value of the mode gap.
        """
        # Defining gap between staves
        gap = np.diff(staves_pos)
        mode,count = stats.mode(gap)
        return int(mode)


    def tone_row_loc(staves_pos):
        """
        Detecting the first row for ach set of tone row.
        Parameters
        ----------------
        :param staves_pos: staves' row coordinate.
        :return: list of the first row on each set of tone row.
        """
        # Defining the first tone line (first coordinate of new tone line)

        tone_row = []
        for i in range(len(staves_pos)-1):
            if i == 0:
                tone_row.append(staves_pos[i])
            else:
                if staves_pos[i] - staves_pos[i-1] >= 3 * staves_gap(staves_pos):
                    tone_row.append(staves_pos[i])
        return tone_row


    def draw_staves(image,staves_pos):
        """
        Draw staves line in an image, assummed to be horizontal.

        Parameters
        ----------------
        :param image: Image to be drawn.
        :param staves_pos: location of each stave.
        :return: Drawn image.
        """
        x0 = 0
        x1 = image.width
        for y in staves_pos:
            draw = ImageDraw.Draw(image)
            draw.line([(x0,y),(x1,y)], fill='red', width=0)
        return image


    def resize_object (object,height,scale):
        """
        Resizing object that wants to be searched.

        Description
        ----------------
        It produces a picture with height the same as the gap of staves and
        keeping the ratio image the same.

        Parameters
        ----------------
        :param object: image to be scaled.
        :param height: the target height of the image (staves gap).
        :param scale: height factor with value >= 0.
        :return: Scaled image.
        """
        ratio = height * scale /object.height
        new_height = int(height * scale)
        new_width = int(object.width * ratio)
        return object.resize(size=(new_width,new_height))


    def CorrelationCoeff(I,R):
        """
        Matching images by finding correlation coefficient between them.

        Parameters
        ----------------
        :param I: input image
        :param R: object to be matched
        :return: Matrix with the same size of the input image with the value [-1,1] where 1 is highly correlated & -1 otherwise
        """
        # Matching functino using Correlation Coefficient Algorithm
        # Ref: Burger & Burge, Principles of Digital Image Processing - Fundamental techniques
        # A window sized of object (R) will be convolved throughout the input image

        # I_ar : input image array
        # R_ar : reference image (object to be matched) array
        I_ar = np.array(ImageOps.invert(I)).astype(np.float64)
        R_ar = np.array(ImageOps.invert(R)).astype(np.float64)

        # Initialization
        K = R.width * R.height
        sum_R = np.sum(R_ar)
        sum_R2 = np.sum(R_ar**2)
        Rbar = sum_R/K
        Sr = np.sqrt(sum_R2 - K * Rbar**2)

        # Generating Correlation Map
        C = np.zeros((I.width-R.width+1, I.height-R.height+1))

        for i in range(I.width-R.width+1):
            for j in range(I.height-R.height+1):
                # Sub image
                sub_I_ar = I_ar[j:j+R.height , i:i+R.width]

                # Compute parameters
                sum_I = np.sum(sub_I_ar)
                sum_I2 = np.sum(sub_I_ar**2)
                sum_IR = np.sum(np.multiply(sub_I_ar,R_ar))

                # Compute correlation
                numerator = sum_IR - sum_I * Rbar
                denom = Sr * np.sqrt(sum_I2 - sum_I**2 / K)

                # Avoiding zero-division
                if denom != 0:
                    C[i][j] = numerator/denom
        return C


    def NonMaximumSuppression(C, teta):
        """
        Removing multiple detection on a single object using non maximum suppression algorithm.

        Description
        ----------------
        Pick a pixels and scan through its neighbors. If it is higher, no changes of value. If it is lower, replace with 0.

        Parameters
        ----------------
        :param C: 2D Correlation Coefficient matrix.
        :param teta: threshold with value [0,1]. Correlation above threshold * max(neighbors) is assumed to be strongly correlated.
        :return: new 2D strongly correlated coefficient matrix.
        """
        # Add paddings for handling edge cases on correlation matrix.
        C_pad = np.pad(C, 3)
        width, height = C_pad.shape

        # Get 3x3 array consists of neighbors and the checked value at the center of the array.
        # Replace the center value with 0.
        # Compare the center value and the max neighbors value (after replacing with 0).
        # If center value < max neighbors, C[i][j] = 0. Otherwise, no changes.

        # Template for new correlation coefficient matrix
        suppressed_C = np.zeros((width, height))

        # Non Maxima Suppression
        for i in range(3, width - 3):
            for j in range(3, height - 3):
                neighbor = np.copy(C_pad[i - 3:i + 4, j - 3:j + 4])
                neighbor[3][3] = 0
                if C_pad[i, j] > np.amax(neighbor) * teta:
                    suppressed_C[i, j] = C_pad[i, j]

        suppressed_C = suppressed_C[3:width - 3, 3:height - 3]
        return suppressed_C


    def BoundingBox (C, teta, size):
        """
        Generating bounding box coordinate.

        Parameters
        ----------------
        :param C: 2D strongly correlated coefficient matrix.
        :param teta: Threshold with value in the range of [0,1]. It decides.
        :param size: tuple of object size (width,height).
        :return: list of coordinate [x1,y1,x2,y2]. x1,y1 is coordiante at the upper left; x2,y2 is lower right.
        """
        width,height = size
        high_cor = np.array(np.transpose(np.where(C >= teta)))
        coordinate = []
        for i in high_cor:
            x1,y1 = i
            x2 = x1 + width
            y2 = y1 + height
            coordinate.append((x1,y1,x2,y2))
        return coordinate


    def drawBoundingBox (image,coordinate,symbol_type):
        """
        Draw bounding box

        Parameters
        ----------------
        :param image: image to be drawn.
        :param coordinate: bounding box corner coordinate (x1,y1,x2,y2).
        :param symbol_type: name of object to be detected (filled note, eight rest, quarter rest).
        :return: new image.
        """
        # This function draws the bounding box from the predicted object coordinate and returns new image
        # The coordinate used is (x1,y1,x2,y2) where x1,y1 is upper left point and x2,y2 is lower right point
        # Object_type : name of object to be searched ('filled_note','eight_rest','quarter_rest')

        # Raise error if symbol_type is not on the list
        type_list = ['filled_note','eight_rest','quarter_rest']
        if symbol_type not in type_list:
            raise ValueError("symbol_type should be one of the following : 'filled_note','eight_rest','quarter_rest'")

        # Color code
        if symbol_type == 'filled_note':
            color = 'red'
        elif symbol_type == 'eight_rest':
            color = 'blue'
        elif symbol_type == 'quarter_rest':
            color = 'lime'

        # Draw bounding box
        for i in coordinate:
            draw = ImageDraw.Draw(image)
            draw.rectangle(list(i), outline=color, width=2)
        return image


    def pitch (tone_row,staves_loc,staves_gap):
        """
        Defining pitches and their location based on the input image.

        Description
        ----------------
        We recognized that the music notes has the '#' and 'b' signs at the beginning which tell us how many pitch
        increase or decrease from the original one.
        In this part, we omit these signs and use the original pitch since our objective is only detecting 3 objects
        of filled note, eight rest, and quarter rest.

        :param tone_row: the first stave on each set of tone row.
        :param staves_loc: the location of staves.
        :param staves_gap: gap between staves.
        :return: dictionary consists of center coordinate as key and name of pitch as value.
        """
        pitch_dict = {}
        for i in tone_row:
            pitch = ['D', 'C', 'B', 'A', 'G', 'F', 'E', 'D', 'C', 'B', 'A', 'G', 'F', 'E', 'D', 'C', 'B']
            if i//staves_gap < 3:
                y1 = i - (i//staves_gap + 0.5) * staves_gap
                sp = (3 - i//staves_gap)*2 - 1
                pitch = pitch[sp:]
            else:
                y1 = i - 3*staves_gap
            for p in pitch:
                y1 += 0.5*staves_gap
                pitch_dict[y1] = p
        return pitch_dict


    def draw_pitch (image,bounding_box,pitch_dict):
        """
        Adding pitch text to the image.

        Parameters
        ----------------
        :param image: image to be drawn
        :param bounding_box: bounding box coordinate from detected filled note
        :param pitch_dict: pitch dictionary consists of {<location> : <pitch_name>}
        :return: new image
        """
        # bounding_box : detected object coordinate (x1,y1,x2,y2)
        # pitch_dict : dictionary consists of pitches as value and their center location in image as key

        # Sort bounding box based on row
        dtype = [('x1', int), ('y1', int), ('x2', int), ('y2', int)]
        bounding_box = np.array(bounding_box, dtype=dtype)
        bounding_box = np.sort(bounding_box, order='y1')

        # Predict pitch on each bounding box
        font = ImageFont.truetype("/Library/Fonts/DejaVuSans.ttf", 14)
        pitch_list = list(pitch_dict.items())
        ix = 0
        result = [] # Lists consists of tuple (row, col, height, width, pitch)
        for x1, y1, x2, y2 in bounding_box:
            yc = (y1 + y2) / 2
            pitch_label = 'NA'
            for i in range(ix,len(pitch_dict)):
                ypitch,pitch = pitch_list[i]
                if yc >= ypitch - 4 and yc <= ypitch + 4:
                    pitch_label = pitch
                    draw = ImageDraw.Draw(image)
                    draw.text((x1-12,yc-7), pitch_label, fill='red', font=font, stroke_width=1)
                    ix = i
                    break
            result.append((x1, y1, y2-y1, x2-x1, pitch_label))
        return image,result

    # -------------------------- Main program --------------------------
    # Load input image which returns binarized color image (for output) and binarized grey image for processing
    input = Image.open(sys.argv[1]).convert(mode='L')
    input = binarize(input)
    c_input = input.convert('RGB')

    # Finding staves lines
    staves_pos = staves(input, input.width * 0.45)  # Threshold for staves line : input.width * 0.45

    # Get the first row of each set of tone row
    first_row = tone_row_loc(staves_pos)

    # Calculate gap between staves
    gap = staves_gap(staves_pos)

    # Get pitch coordinate dictionary
    pitch_dict = pitch(first_row,staves_pos,gap)


    # Load and Resizing object based on staves gap
    note = Image.open('template1.png').convert(mode='L')
    resized_note = resize_object(note, gap, 1)
    resized_note = binarize(resized_note)
    # Add padding to note as the template image slightly cropped resuilting in small shifting in boundary box location
    # This will help the model to find almost full part of note.
    resized_note = padding(resized_note,(1,0))

    q_rest = Image.open('template2.png').convert(mode='L')
    resized_q_rest = resize_object(q_rest,gap,3)
    resized_q_rest = binarize(resized_q_rest)

    e_rest = Image.open('template3.png').convert(mode='L')
    resized_e_rest = resize_object(e_rest,gap,2)
    resized_e_rest = binarize(resized_e_rest)


    # Creating list of symbols to be searched
    symbol_list = [(resized_note,'filled_note'), (resized_e_rest,'eight_rest'), (resized_q_rest,'quarter_rest')]

    # Matching symbol
    result_table = []
    teta1 = 0.58   # Threshold for matching symbol
    for symbol_tuple in symbol_list:
        # Matching object
        symbol_img, name = symbol_tuple
        C = CorrelationCoeff(input,symbol_img)
        C = NonMaximumSuppression(C,1)
        bounding_box = BoundingBox(C,teta1,(symbol_img.width,symbol_img.height))
        scanned_img = drawBoundingBox(c_input,bounding_box,name)

        # Finding pitch
        if name == 'filled_note':
            detected_img, note = draw_pitch(scanned_img,bounding_box,pitch_dict)

            # Append deteted object result to the result_table list
            dtype = [('row', int), ('col', int), ('height', int), ('width', int), ('pitch', object)]
            note = np.array(note, dtype=dtype)
            note = np.array(note, dtype=dtype)
            note = np.sort(note, order=['row', 'col'])
            for row, col, height, width, pitch_label in note:
                result_table.append([row, col, height, width, name, pitch_label, '1'])
        else:
            for x1,y1,x2,y2 in bounding_box:
                result_table.append(([x1,y1,y2-y1,x2-x1,name,'_','1']))


    # Saving image in the repository
    scanned_img.save('detected.png')

    # -------------------------- Table --------------------------
    header = ['<row>', '<col>', '<height>', '<width>', '<symbol_type>', '<pitch>', '<confidence>']
    format = '{:18}' * len(header)
    print(format.format(*header))
    print('=' * len(format) * 4)
    for i in result_table:
        print(format.format(*map(str,i)))

