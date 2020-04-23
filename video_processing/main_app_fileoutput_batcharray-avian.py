import math
import cv2
import copy
import os
import sys
import time
import argparse
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

def find_objects(path, file, filename, outpath, showImages, writeImages):
    cap = cv2.VideoCapture(os.path.join(path, file))
    storeImages = True
    #writeImages = True
    #showImages = False

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
    

def read_files(path, outpath, show_video, write_images):
    for filePath in sorted(os.listdir(path)):
        file = os.path.join(path, filePath)
        out_file.write("Attempting to process file: " + file + "\n")
        out_file.flush()
        if os.path.isdir(file):
            read_files(file, outpath)        
        splitPath = os.path.splitext(filePath)
        fileName = splitPath[0]
        fileExt = splitPath[1]
        if fileExt in [".mp4", ".MP4", ".MOD", "mod", ".MTS", ".mts", ".mkv"]:
            start = time.time()
            out_file.write("File valid, starting processing\n")
            out_file.flush()
            find_objects(path, file, fileName, outpath, show_video, write_images)
            delta = time.time() - start
            out_file.write("Took: " + str(delta) + " seconds to process video\n")
            out_file.flush()
    out_file.write("Done with all files in directory\n")
    out_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirlist_file", help="The file with a list of the sub directories to process")
    parser.add_argument("videos_dir", help="The path to the top level folder where the list of video directories are")
    parser.add_argument("output_dir", help="The path to the output folder to use if writing out moving object images")
    parser.add_argument("dir_index", type=int, help="The index of the directory of videos to process. Matches a row in the dirlist file")
    parser.add_argument("-show_video", action="store_true", help="Should the video be displayed while processing?")
    parser.add_argument("-write_images", action="store_true", help="Should images of the moving objects be written to disk?")

    args = parser.parse_args()
    out_filename = os.path.join(args.output_dir, "processing_log.txt")
    with open(out_filename, 'w') as out_file:
        arg = args.dir_index
        out_file.write("using file index " + str(arg) + "\n")
        out_file.flush()
        params_file = pd.read_csv(args.dirlist_file)
        params = params_file.loc[params_file['Index'] == arg]
        out_file.write("show_video = " + str(args.show_video) + "\n")
        out_file.write("write_images = " + str(args.write_images) + "\n")
        out_file.flush()
        if (params.size > 0):
            target = params['Directory'].iloc[0]
            path = os.path.join(args.videos_dir, target)
            outpath = os.path.join(args.output_dir, target)
            read_files(path, outpath, args.show_video, args.write_images)
        out_file.write("Done all\n")
        out_file.flush()
