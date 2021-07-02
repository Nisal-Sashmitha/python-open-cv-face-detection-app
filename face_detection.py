


import cv2

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


#choose an  image to detect face in
#img =  cv2.imread('RD3.jpg')
webcam = cv2.VideoCapture(0)



#iterate forever over frames
while True:
    #read current frame
    successful_frame_read, frame = webcam.read()


    #must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    print(face_coordinates)

    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0), 2)


    cv2.imshow('NIKI FACE DETECTOR',frame)

    
    if cv2.waitKey(10) ==ord('a'):
        break

webcam.release()    
print("code completed")    





