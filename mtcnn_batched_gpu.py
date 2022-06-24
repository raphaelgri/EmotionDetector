from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import math
#from tqdm.notebook import tqdm

#initialize the model, selecting GPU as device and keeping all faces
mtcnn = MTCNN(select_largest=False, device='cuda', keep_all=True,post_process=False)

#Load the video
v_cap = cv2.VideoCapture('Data/videoplayback.mp4')
v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(v_cap.get(cv2.CAP_PROP_FPS))
timestamps = [] #list for the timestamps

# Loop through video
batch_size = 64
batch_num = 0
frames = []
boxes = []
landmarks = []
view_frames = []
view_boxes = []
view_landmarks = []
sf = 1 # get one frame per second

for i in range(v_len):
    # Load frame
    success, frame = v_cap.read()
    #select every n frames
    if i % math.floor(fps)*sf == 0: #skip to get every sf seconds
        success, frame = v_cap.retrieve()
    else:
        continue
    if not success:
        continue

    # Add to batch, resizing for speed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = frame.resize([int(f * 0.25) for f in frame.size])
    frames.append(frame)
    timestamps.append(math.floor(v_cap.get(cv2.CAP_PROP_POS_MSEC))*1000) #append timestamp of the frame
    print(math.floor(v_cap.get(cv2.CAP_PROP_POS_MSEC))*1000)
    
    # When batch is full, detect faces and reset batch list
    if len(frames) >= batch_size:
        #print("Executed MTCNN batch num" + str(batch_num))
        #define save path 
        save_paths = [f'Data/T2/image_t{timestamps[i]}.jpg' for i in range(len(frames))]
        #save all faces to files
        faces = mtcnn(frames, save_path=save_paths)
        im_count = 0
        #increase batch number
        batch_num += 1
        frames = [] #clear frame list
        timestamps = [] #clear timestamplist