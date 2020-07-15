import cv2
import numpy as np
import dlib
import math
from PIL import Image
import glob


image_list = []
for filename in glob.glob('zdjecia/*.*'):
    im = cv2.imread(filename)
    image_list.append(im)
#for filename in glob.glob ('')
image = cv2.imread("kask.png", -1)
i = "c"
name = "dupa.jpg"

for image2 in image_list:
    frame = np.zeros((image2.shape[0] + 120, image2.shape[1],3), dtype = "uint8")
    frame[120 : image2.shape[0] + 120, 0 : image2.shape[1]] = image2
    kx = image.shape[1]
    ky = image.shape[0]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #frame = frame[frame.shape[0] : 0,:]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1_x = 0
    p1_y = 0
    p2_x = 0
    p2_y = 0
    nos_x = 0
    nos_y = 0
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(gray, face)
        p1_x = landmarks.part(19).x
        p1_y = landmarks.part(19).y
        p2_x = landmarks.part(33).x
        p2_y = landmarks.part(33).y

        nos_x = landmarks.part(27).x
        nos_y = landmarks.part(27).y

        ratio = (abs(landmarks.part(0).x - landmarks.part(16).x) + 50) / kx
        new_size = (abs(landmarks.part(0).x - landmarks.part(16).x) + 50 ,int(ratio * ky))
        image2 = cv2.resize(image, new_size);

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)


        dist = int((math.sqrt(( p1_x - p2_x) **2 + ( p1_y - p2_y) ** 2))/2)
        maxy = frame.shape[0]
        maxx = frame.shape[1]

        f1x = int(nos_y - 0.5 * dist - image2.shape[0])
        f2x = int(nos_y - 0.5 * dist)
        f1y = int(nos_x - image2.shape[1]/2 - 0.05 * dist + image2.shape[1])
        f2y = int(nos_x - image2.shape[1]/2 - 0.05 * dist)
        min = 0
        name = i + name


        alpha_s = image2[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s



        if f1x > 0 and f2x < maxy and f2y > 0 and f1y< maxx:
            for c in range(0, 3):
                frame[f1x:f2x, f2y:f1y, c] = (alpha_s * image2[:, :, c] + alpha_l * frame[f1x:f2x, f2y:f2y, c])
            #image2 = cv2.addWeighted(frame[f1x: f2x, f2y : f1y],0.4, image2, 0.1, 0)
            #frame[f1x : f2x, f2y : f1y] = image2
                #y                      x
        cv2.imshow("Frame", frame)
        cv2.imwrite(name, frame)

    key = cv2.waitKey(1)
    if key == 27:
        break