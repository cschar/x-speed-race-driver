import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean

from PIL import ImageGrab
from mss import mss
from PIL import Image
import cv2
import time
import pyautogui

# from pylab import array, plot, show, axis, arange, figure, uint8 

#doesnt work
def brighten_image(img):
    
    maxIntensity = 255.0 # depends on dtype of image data
    x = np.arange(maxIntensity) 

    # Parameters for manipulating image data
    phi = 1
    theta = 1

    # Increase intensity such that
    # dark pixels become much brighter, 
    # bright pixels become slightly bright
    newImage0 = (maxIntensity/phi)*(img/(maxIntensity/theta))**2
    newImage0 = np.array(newImage0,dtype=np.uint8)
    return newImage0

def roi(img, vertices):
    mask = np.zeros_like(img) # make big zero array in img dimensions
    cv2.fillPoly(mask, vertices, 255) # fill in mask @ vertices w 255 value
    masked = cv2.bitwise_and(img, mask)
    return masked

p_mode = 6
bright = False
def process_img(image):
    ''' image is a numpy array '''
    global p_mode
    global bright
    m1 = 0
    m2 = 0
    t1 = 50
    t2 = 200
    primg = image
    original_image = image
    if bright:
        primg = brighten_image(primg)
        print("brighten")

    # if p_mode == 1:
    #     # primg = cv2.cvtColor(primg, cv2.COLOR_BGR2RGB)
    #     print('bgr 2 rgb')
    if p_mode >= 2:
        # print('--> Gray',)

        # primg = cv2.cvtColor(primg, cv2.COLOR_BGR2RGB)
        primg = cv2.cvtColor(primg, cv2.COLOR_RGB2GRAY)
    if p_mode >= 3:
        # edge detect
        # print(' --> Edge',)
        # primg = cv2.cvtColor(primg, cv2.COLOR_BGR2RGB)
        # primg = cv2.cvtColor(primg, cv2.COLOR_RGB2GRAY)    
        primg =  cv2.Canny(primg, threshold1 = t1, threshold2=t2)

    if p_mode >= 4:
        # print('ROI Mask (blur)',)
        
        
        # vertices = np.array([[10, 500], [10, 300],
        #  [300, 200], [500,200], [800,300], [800, 500]])
        #optional blur
        primg = cv2.GaussianBlur(primg, (5,5), 0)

        vertices = np.array([ 
          [10, 500], [10, 400],  # left side
         [300, 300], [500,300],  # middle 
          [800,400], [800, 500]])  # right side
        primg = roi(primg, [vertices])

    if p_mode >= 5:
        # print('--> HoughLine',)
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
        minLineLength = 100  # 20
        maxLineGap = 5
        #                       edges
        lines = cv2.HoughLinesP(primg, 1, np.pi/180,
                                180, np.array([]), minLineLength, maxLineGap)
        draw_lines(primg, lines)
        
        
    if p_mode >= 6:
        # print('--> Lanes',)
        result = draw_lanes(original_image,lines)
        if result is not None:
            l1, l2, m1, m2 = result
            try:
                cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
                cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
            # import ipdb; ipdb.set_trace();
            except Exception as e:
                print('draw_lanes cv2.line exception')
                print(str(e))
    
    return primg, original_image, m1, m2




def draw_lines(primg, lines):
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(primg, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)       
            except Exception as e:
                print('draw hough lines exception')
                print(str(e))
    except Exception as e:
        pass

def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):

    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker 
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []  
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]]
        min_y = min(ys)
        max_y = 600
        line_dict = {}

        # make new lines, running from min_y to max_y
        # max_y being, bottom of screen,
        # min_y being top most 'y' position of all lines generated
        # by find edges/ houg lines
        for idx,i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]
                if abs(m) < 0.0000001: # if its zero
                    # print('found dangerously SMALL m, skipping...')
                    continue
                # Calculating our new, and improved, xs
                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        # ids are the slopes
        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return ([l1_x1, l1_y1, l1_x2, l1_y2],
         [l2_x1, l2_y1, l2_x2, l2_y2], lane1_id, lane2_id)
    except Exception as e:
        print(str(e))


# from direct_keys import straight, right, left

def straight():
    pyautogui.keyDown('up')
    time.sleep(0.20)
    pyautogui.keyUp('up')

def left():
    pyautogui.keyDown('up')
    pyautogui.keyDown('left')
    time.sleep(0.10)
    pyautogui.keyUp('up')
    time.sleep(0.25)
    pyautogui.keyUp('left')

def right():
    pyautogui.keyDown('up')
    pyautogui.keyDown('right')
    time.sleep(0.10)
    pyautogui.keyUp('up')
    time.sleep(0.25)
    pyautogui.keyUp('right')


def screen_record():
    global p_mode
    global bright
    drive = False
    drive = True

    if drive:
        for i in range(2)[::-1]:
            print(i+1)
            time.sleep(1)

    # box = { "top": 45, "left": 0, "width": 480, "height": 320}
    box = { "top": 45, "left": 0, "width": 800, "height": 600}
    sct = mss()

    processed_window_name = '=Processed Image'
    original_window_name = '=Original Image'

    cv2.namedWindow(processed_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(original_window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(processed_window_name, x=500, y= -800)
    cv2.moveWindow(original_window_name, x=-200, y= -800)

    drive_time = time.time()
    from collections import defaultdict
    drive_votes = defaultdict(int)
    while True:
        t = time.time()
        sct.get_pixels(box)
        image = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        # Could be useful in MAC OS    
        b, g, r = image.split()
        image = Image.merge("RGB", (r, g, b))

        np_image = np.array(image)
        primg, original_image, m1, m2 = process_img(np_image)

        cv2.imshow('=Processed Image', primg)
        cv2.imshow('=Original Image', original_image)
        
        
        if m1 < 0 and m2 < 0:
            direction = right
        elif m1 > 0 and m2 > 0:
            direction = left
        else:
            direction = straight

        drive_votes[direction] += 1

        if drive:
            for key, value in drive_votes.items():
                print("drive vote {} --> {}".format(key.__name__, value))
            winner = max(drive_votes, key=drive_votes.get)  
            if (t - drive_time) > 2.5:  # drive every 2 seconds
                print('driving! {}'.format(winner.__name__))
                # direction()
                winner()
                drive_time = time.time()
                drive_votes = defaultdict(int)

        print('[{:10}] m1 {:6.4f}  m2 {:6.4f}'.format(direction.__name__, m1, m2))


        # print('fps: {}'.format(1/(time.time()-t)))
        key = cv2.waitKey(25)
        if key & 0xFF == ord('q'):  # wait 25 ms
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord('b'):    
            bright = not bright
        if key & 0xFF == ord('1'):
            p_mode = 1
            print(1)
        if key & 0xFF == ord('2'):
            p_mode = 2
        if key & 0xFF == ord('3'):
            p_mode = 3
        if key & 0xFF == ord('4'):    
            p_mode = 4
        if key & 0xFF == ord('5'):    
            p_mode = 5
        if key & 0xFF == ord('6'):    
            p_mode = 6


def test_process():
    box = { "top": 45, "left": 0, "width": 800, "height": 600}
    sct = mss()
    global p_mode

    
    t = time.time()
    sct.get_pixels(box)
    # image = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
      # Could be useful in MAC OS    
    # b, g, r = image.split()
    # image = Image.merge("RGB", (r, g, b))

    # cv.imread doesnt seem to need to reshuffle rgb
    # image = cv2.imread('/Users/codyscharfe/Desktop/xpspeedtest.png')
    # image = cv2.imread('/Users/codyscharfe/Desktop/xspr2test.png')
    image = cv2.imread('/Users/codyscharfe/Desktop/float2int.png')
  

    np_image = np.array(image)
    p_mode = 6

    primg, original_image = process_img(np_image)

    cv2.imshow('=Processed Image', primg)
    cv2.imshow('=Original Image', original_image)
    cv2.moveWindow('=Original Image', x=500, y= -800)
    cv2.moveWindow('=Processed Image', x=0, y= -800)
    print('fps: {}'.format(1/(time.time()-t)))

    while True:
        key = cv2.waitKey(25)
        if key & 0xFF == ord('q'):  # wait 25 ms
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test_process()
    else:
        screen_record()

