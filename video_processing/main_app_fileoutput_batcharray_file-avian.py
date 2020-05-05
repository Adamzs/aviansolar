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

def find_objects(file, filename, outpath, showImages, writeImages):
    cap = cv2.VideoCapture(file)
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
        print('processed frame: ' + str(count))# + " frame size = " + str(len(frame)))
        count += 1
    cap.release()
    

def read_files(file, fileName, fileExt, outpath, show_video, write_images):
    out_file.write("Attempting to process file: " + file + "\n")
    out_file.flush()
    if fileExt in [".mp4", ".MP4", ".MOD", "mod", ".MTS", ".mts", ".mkv"]:
        start = time.time()
        out_file.write("File valid, starting processing\n")
        out_file.flush()
        print("file = " + file)
        print("fileName = " + fileName)
        print("outpath = " + outpath)
        find_objects(file, fileName, outpath, show_video, write_images)
        delta = time.time() - start
        out_file.write("Took: " + str(delta) + " seconds to process video\n")
        out_file.flush()
    out_file.write("Done with all files in directory\n")
    out_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filelist_file", help="The file with a list of the files in sub directories to process")
    parser.add_argument("videos_dir", help="The path to the top level folder where the list of video directories are")
    parser.add_argument("output_dir", help="The path to the output folder to use if writing out moving object images")
    parser.add_argument("dir_index", type=int, help="The index of the video file to process. Matches a row in the filelist file")
    parser.add_argument("-show_video", action="store_true", help="Should the video be displayed while processing?")
    parser.add_argument("-write_images", action="store_true", help="Should images of the moving objects be written to disk?")

    args = parser.parse_args()
    arg = args.dir_index
    print("using file index " + str(arg) + "\n")

    params_file = pd.read_csv(args.filelist_file)
    params = params_file.loc[params_file['Index'] == arg]
    if (params.size > 0):
        rel_video_file = params['Directory'].iloc[0]
        print("rel video file = " + rel_video_file)
        full_path_video_file = os.path.join(args.videos_dir, rel_video_file)
        print("full path video file = " + full_path_video_file)
        rel_outdir = os.path.dirname(rel_video_file)
        full_path_output_dir = os.path.join(args.output_dir, rel_outdir)
        print("output directory = " + full_path_output_dir)
        splitPath = os.path.splitext(full_path_video_file)
        fileName = os.path.basename(splitPath[0])
        print("fileName = " + fileName)
        fileExt = splitPath[1]
        out_path = os.path.join(full_path_output_dir, fileName)
        if not os.path.exists(out_path):
            try:
                os.makedirs(out_path)
            except FileExistsError:
                pass
            
        out_filename = os.path.join(out_path, "processing_log.txt")
        with open(out_filename, 'w') as out_file:
            read_files(full_path_video_file, fileName, fileExt, full_path_output_dir, args.show_video, args.write_images)
