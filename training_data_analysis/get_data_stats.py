# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:53:29 2020

@author: szymanski
"""
import os
import argparse

def getTrackInfo(video_dir):
    speed = 0
    area = 0
    count = 0
    for image_file in sorted(os.listdir(video_dir)):
        splitPath = os.path.splitext(image_file)
        fileName = splitPath[0]
        fileExt = splitPath[1]
        if fileExt in [".png",".jpg"]:
            vals = fileName.split('_')
            speed += float(vals[9])
            area += int(vals[10])
            count += 1
    return [count, (speed / count), (area / count)]

def processCameraDirectory(camera_dir, path, pretty_print, details, out_file):
    # there should be a subdirectory for each video
    # and many subdirectories under that for each track
    # in each track subdirectory there will be a collection of .png files
    # in the png file name there will be the size and speed of the object
    camera_path = os.path.join(path, camera_dir)
    numvideos = 0
    file_lines = []
    if pretty_print:
        print("   " + camera_dir + " contains:");
    path_str = path + "," + camera_dir
    for video_dir in sorted(os.listdir(camera_path)):
        video_path = os.path.join(camera_path, video_dir)
        if os.path.isdir(video_path):
            if pretty_print:
                print("      " + video_dir + " contains:")
            file_str = path_str + "," + video_dir
            numvideos += 1
            trackcount = 0
            lines = []
            lines_csv = []
            for track_dir in sorted(os.listdir(video_path)):
                track_path = os.path.join(video_path, track_dir)
                if os.path.isdir(track_path):
                    trackcount += 1
                    if details:
                        track_info = getTrackInfo(track_path)
                        lines.append("         Track " + track_dir + ": Length: " + str(track_info[0]) + ", Ave Size: " + str(round(track_info[2], 1)) + ", Ave Speed: " + str(round(track_info[1], 2)))
                        lines_csv.append("," + track_dir + "," + str(track_info[0]) + "," + str(round(track_info[2], 1)) + "," + str(round(track_info[1], 2)))

            if pretty_print:
                print("         Number of Tracks: " + str(trackcount))
                if len(lines) > 0:
                    print("\n".join(lines))
                    
            file_str = file_str + "," + str(trackcount)
            if details:
                for row in lines_csv:
                    file_lines.append(file_str + row)
            else:
                file_lines.append(file_str)
                    
    for line in file_lines:
        print(line)
        out_file.write(line + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="The directory to process")
    parser.add_argument("-print", action="store_true", help="Pretty print to console")
    parser.add_argument("-details", action="store_true", help="Should the track details be shown")

    args = parser.parse_args()
    path = args.directory
    if args.print:
        print("Creating statistics for dir " + str(path))
        
    out_filename = os.path.join(path, "data_stats.csv")
    with open(out_filename, 'w') as out_file:
        if args.details:
            out_file.write("directory,camera,video_dir,num_tracks,track_dir,track_len,ave_size,ave_speed\n")
        else:
            out_file.write("directory,camera,video_dir,num_tracks\n")
            
        for filePath in sorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path, filePath)):
                processCameraDirectory(filePath, path, args.print, args.details, out_file)

        out_file.flush()
        out_file.close()
