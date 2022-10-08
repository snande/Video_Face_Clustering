import streamlit as st
import numpy as np
import cv2
import face_recognition as fr
from PIL import Image
import os
from pytube import YouTube
import shutil

input_type = st.radio("Input Selector", ('FILE', 'URL'))

if input_type == 'FILE':
    uploaded_file = st.file_uploader("Upload a Video < 200MB")
        
elif input_type == 'URL':
    url = st.text_input('Link to a YouTube Video', "")
    if url != "":
        yt = YouTube(url)
        uploaded_file = yt.streams.get_by_resolution("360p").download('videos')
        print("Video Downloaded Successfully!")
    else:
        uploaded_file = None
             

if uploaded_file is not None:
    
    count = 0
    face_num = 0
    enc_list = []
    try:
        os.mkdir('images')
    except:
        pass
    cap = cv2.VideoCapture(uploaded_file)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    multiplier = int(np.sqrt(total_frames))
    
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, count*multiplier)
            img = frame[:,:,::-1]
            face_locs = fr.face_locations(img)

            if len(face_locs)>0:
                for j in range(len(face_locs)):
                    enc = fr.face_encodings(img, known_face_locations=[face_locs[j]])
                    enc_list.append(enc[0])
                    top, right, bottom, left = face_locs[j]
                    length = (right-left)
                    extra = int(length*15/100)
                    if (top-int((extra)*3.5)) >= 0:
                        topn = (top-int((extra)*3.5))
                    else:
                        topn = 0
                    if left-extra >= 0:
                        leftn = left-extra
                    else:
                        leftn = 0
                    img_array = img[topn:bottom+extra,leftn:right+extra,:]
                    Image.fromarray(img_array).resize((200,200)).save('images/face{}.jpg'.format(face_num))
                    face_num += 1
            else:
                continue

        else:
            cap.release()
            break
    
    group_dict = {}
    for i in range(len(enc_list)):
        group_dict['face{}'.format(i)] = 'person{}'.format(i)
        
    for i in range(len(enc_list)-1):
        dist = fr.face_distance(enc_list[i+1:], enc_list[i])
        for j in range(len(dist)):
            if dist[j]<=0.55:
                group_dict['face{}'.format(i+j+1)] = group_dict['face{}'.format(i)]
                
    for i in set(group_dict.values()):
        grouplist = ['images/{}.jpg'.format(j) for j in group_dict.keys() if group_dict[j]==i]
        st.text('Images of : ' + i)
        st.image(grouplist[::len(grouplist)//6 if len(grouplist)>6 else 1][:6])
        
    try:
        shutil.rmtree('images', ignore_errors=True)
        shutil.rmtree('videos', ignore_errors=True)
    except:
        pass