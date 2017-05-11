
from os import system
import time
import pyautogui
# left = 123
# right = 124
# down = 125
# up = 126

# def drive_car(arrow_key_number):
#     print('driving {}'.format(arrow_key_number))
#     file_location = '/tmp/drive_car_script.as'
#     applescript = """
# activate application "XSpeedRace2"
# tell application "System Events"
#         -- works!
#         repeat 30 times
#             key code {}
#             delay 0.01 
#         end
# end
# """.format(arrow_key_number)

#     f = open(file_location, 'w')
#     f.write(applescript)
#     f.close()

#     system("osascript {}".format(file_location))
    

# def straight():
#     drive_car(up)

# def left():
#     drive_car(left)

# def right():
#     drive_car(right)

def straight():
    pyautogui.keyDown('up')
    time.sleep(0.5)
    pyautogui.keyUp('up')

def left():
    pyautogui.keyDown('left')
    time.sleep(0.5)
    pyautogui.keyUp('left')

def right():
    pyautogui.keyDown('right')
    time.sleep(0.5)
    pyautogui.keyUp('right')

if __name__ == '__main__':
    
    for i in range(3)[::-1]:
        print(i+1)
        time.sleep(1)

    straight()
    right()
    left()
    left()

    

    