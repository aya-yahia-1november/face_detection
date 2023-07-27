import  cv2
import  numpy as np

eye_cascade =cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture("istockphoto-1130631457-640_adpp_is.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, (250, 250))
while cap.isOpened():
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        cv2.putText(frame,"face",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,1,(12,34,3),1)
        """
        roi_gray=gray[y:y+h,x:x+w]
        roi_colored=frame[y:y+h,x:x+w]
        """
        eyes=eye_cascade.detectMultiScale(gray,2.3,4)
        for (ex,ey,ew,eh) in eyes:
          cv2.rectangle(frame,(ex,ey),(ex+ew,eh+ey),(0,234,255),2)
          cv2.putText(frame, "EYE", (ex, ey-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (12, 34, 3), 1)
    cv2.imshow("frame",frame)
    output.write(frame)
    if cv2.waitKey(1)==ord('q'):
            cv2.destroyWindow()
            break
cap.release()
cv2.destroyWindow()
output.release()