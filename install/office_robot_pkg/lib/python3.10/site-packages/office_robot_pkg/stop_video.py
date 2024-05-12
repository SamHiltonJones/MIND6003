#!/usr/bin/env python3
import av
import cv2
import os
import traceback

def create_video(path, fps, output):
    vid = av.open(output, "w")
    vs = vid.add_stream("libx264", rate=fps)
    vs.pix_fmt = 'yuv420p'
    vs.bit_rate = 1024000 

    frame_index = 0
    for subdir, dirs, files in os.walk(path):
        files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        files_to_process = files[2:] if len(files) > 2 else []
        for file in files_to_process:
            filepath = os.path.join(subdir, file)
            try:
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img is not None and img.shape[2] == 4: 
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                if img is not None:
                    new_frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                    new_frame.pts = frame_index
                    frame_index += 1
                    packet = vs.encode(new_frame)
                    vid.mux(packet)
            except Exception as e:
                print(e)
                traceback.print_exc()

    for packet in vs.encode():
        vid.mux(packet)

    vid.close()

create_video('/home/sam/tut/MIND6003/frames', 15, '/home/sam/tut/MIND6003/video.avi')
