import math
import cv2
import copy
import os
import sys
import time
import pandas as pd
from obj_detector import ObjDetector
from tracker import Tracker

def calc_trajectory(track):
    if len(track.trace) > 5:
        x1 = track.trace[-5][0]
        y1 = -track.trace[-5][1]
        x2 = track.trace[-1][0]
        y2 = -track.trace[-1][1]
        dx = x2 - x1
        dy = y2 - y1
        return calc_angle(dx, dy)
    return 0

def calc_angle(dx, dy):
    rads = math.atan2(dy, dx)
    degs = math.degrees(rads)
    adj = 360 - (degs - 90)
    if adj >= 360:
        return adj - 360
    return adj        

def find_objects(path, file, filename, outpath):
    cap = cv2.VideoCapture(os.path.join(path, file))
    storeImages = True
    writeImages = True
    showImages = False

    detector = ObjDetector(showImages, 200)
    tracker = Tracker(900, 7, 150, 100, storeImages, writeImages, filename, outpath)
    
    if showImages == True:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800, 800)

    count = 1
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break

        orig_frame = copy.copy(frame)

        centers = detector.findObjects(frame)

        tracker.updateTracks(centers, orig_frame, count)
        #print('processed frame: ' + str(count))# + " frame size = " + str(len(frame)))
        count += 1
    cap.release()
    

def read_files(path, outpath):
    for filePath in sorted(os.listdir(path)):
        file = os.path.join(path, filePath)
        print("Attempting to process file: " + file)
        if os.path.isdir(file):
            read_files(file, outpath)        
        splitPath = os.path.splitext(filePath)
        fileName = splitPath[0]
        fileExt = splitPath[1]
        if fileExt in [".mp4", ".MP4", ".MOD", "mod", ".MTS", ".mts", ".mkv"]:
            #imagePath = os.path.join(path, filePath)
            start = time.process_time()
            print("File valid, starting processing")
            find_objects(path, file, fileName, outpath)
            delta = time.process_time() - start
            print("Took: " + str(delta) + " seconds to process video")
            #print(imagePath)
    print("Done with all files in directory")


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("no param index specified, exiting")
    else:
        arg = int(sys.argv[1])
        print("using file index " + str(arg))
        #params_file = pd.read_csv("/project/msca/projects/dspotter/uasvideos/dirs_list.csv")
        params_file = pd.read_csv("D:\\AvianSolarData\\videos\\dirs_list.csv")
        params = params_file.loc[params_file['Index'] == arg]

        if (params.size > 0):
            target = params['Directory'].iloc[0]
            #path = os.path.join('/project/msca/projects/dspotter/uasvideos', target)
            path = os.path.join('D:\\AvianSolarData\\videos', target)
            #outpath = os.path.join('/scratch/midway2/azs/image-output', target)
            outpath = os.path.join('D:\\AvianSolarData\\videos\\image_output', target)
            read_files(path, outpath)
        print("Done all")