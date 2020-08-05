import cv2

img = cv2.VideoCapture('pedestrian.mp4')

car_tracker = cv2.CascadeClassifier('cars.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbodyy.xml')

while True:
    read_successful,frame = img.read()

    if read_successful:
        gray_scaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    cars = car_tracker.detectMultiScale(gray_scaled)
    pedestrians = pedestrian_tracker.detectMultiScale(gray_scaled)

    for(x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255),2)

    cv2.imshow('AI CAR DETECTION',frame)

    if cv2.waitKey(1) == 27:
        break

