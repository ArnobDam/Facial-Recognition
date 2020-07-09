import sys
import os
import cv2
import face_recognition
import imutils
import pandas as pd
import time

path = '/Users/Z.Steinberg/Documents/Current Classes/Capstone/capstone/Facial Recognition/testing/vids'
movie = 'yamiche.mp4'

known_pics = os.listdir('/Users/Z.Steinberg/Documents/Current Classes/Capstone/capstone/Facial Recognition/testing/known_pics')
pic_path = '/Users/Z.Steinberg/Documents/Current Classes/Capstone/capstone/Facial Recognition/testing/known_pics'

start_time = time.time()
vid_path = os.path.join(path,movie)
input_movie = cv2.VideoCapture(vid_path)

# Encode Known Pictures
encodings = []
names = []
for pic in known_pics:
	img = face_recognition.load_image_file(os.path.join(pic_path,pic))
	enc = face_recognition.face_encodings(img)[0]
	name = pic[:-4]
	encodings.append(enc)
	names.append(name)
print("encodings finished")

face_info = {}
for name in names:
    face_info[name] = []
face_info['Match'] = []
face_info['Time'] = []
print(face_info)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
#     Get Frame
    frame_time = time.time()
    ret, frame = input_movie.read()
    
#     Quit when movie ends
    if not ret:
        break
        
#         Convert to RGB
    frame = frame[:,:,::-1]
    # frame = imutils.rotate(frame,-90)
    frame = imutils.resize(frame, width=400, height=400)
    
#     Locations and encodings
    locations = face_recognition.face_locations(frame, model='cnn')
    face_encs = face_recognition.face_encodings(frame,locations)
    
    # For each face encoding found in the video input
    for enc in face_encs:
        match = face_recognition.compare_faces(encodings,enc, tolerance=0.50)
        distances = face_recognition.face_distance(encodings,enc)
        
        # Count the number of matches
        num_times = match.count(True)
        print('Num Matches: ' + str(num_times))
        
        if True in match: #If there is a match
            name = names[match.index(True)]

            print('Matched with: '+str(name))
            for i in range(len(distances)):
                face_info[names[i]].append(distances[i])    # Add distance from known image to face_info dictionary
            face_info['Match'].append(name)                 # Add Match Name
            face_info['Time'].append(time.time()-frame_time)# Add time per frame
        else: #No Match
            print('no match')
            for i in range(len(distances)):
                face_info[names[i]].append(distances[i])    # Add distance from known image to face_info dictionary
            face_info['Match'].append("No Match")           # Add that there was no match
            face_info['Time'].append(time.time()-frame_time)# Add time per frame

# Write info to csv file for processing
df = pd.DataFrame.from_dict(face_info)
df.to_csv('csv/yamicheCNN.csv')
print("--- %s seconds ---" % (time.time() - start_time))

# When Finished
input_movie.release()
cv2.destroyAllWindows()