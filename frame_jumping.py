import cv2

path = str(input)
(x1, y1) = list(map(int, input("Please enter the left side coordinates for the line: ").split(" "))) #space separated 2 integers
(x2, y2) = list(map(int, input("Please enter the right side coordinates for the line: ").split(" ")))

CaptureVideo = cv2.VideoCapture(r"/home/aditya/Desktop/ML+CV/1615363610851.mp4")
CaptureVideo = cv2.VideoCapture(path)

# BS_KNN = cv2.createBackgroundSubtractorKNN()    
BS_MOG2 = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

vehicle_count = 0
while CaptureVideo.isOpened():
    ret, f = CaptureVideo.read()
    height, width, _ = f.shape
#     print(height, width)
   
    
    
    target_area = f[y1-250: y2+250, x1: x2+50]
    ForeGroundMask = BS_MOG2.apply(target_area)
    _, ForeGroundMask = cv2.threshold(ForeGroundMask, 254, 255, cv2.THRESH_BINARY)
#     cv2.line(f, (0, 450), (1500, 450), (0,0,255), 3) #red line
#     cv2.line(f, (0, 440), (1500, 440), (255,0,0), 1) #blue top offset line
#     cv2.line(f, (0, 460), (1500, 460), (255,0,0), 1) #blue bottom offset line

    cv2.line(f, (x1, y1), (x2, y2), (0,0,255), 3) #red line
    cv2.line(f, (x1, y1-10), (x2, y2-10), (255,0,0), 1.5) #blue top offset line
    cv2.line(f, (x1, y1+10), (x2, y2+10), (255,0,0), 1.5) #blue bottom offset line
  
    contours, NaN = cv2.findContours(ForeGroundMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        if (cv2.contourArea(i)<1900):
            continue
        x,y,w,h = cv2.boundingRect(i)
        cv2.rectangle(f, (x,y), (x+w, y+h), (0,255,0), 2)
        xCentre = int((x + (x+w))/2)
        yCentre = int((y + (y+h))/2)
        cv2.circle(f, (xCentre, yCentre), 5, (0,0, 255), 5)
        if yCentre>440 and yCentre<460:
            vehicle_count+=1


    cv2.imshow("Target area", target_area)
    cv2.imshow("Foreground Mask", ForeGroundMask)
    cv2.putText(f, "Total Vehicles: {}".format(vehicle_count), (450, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5)
    cv2.imshow("Original video", f)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
CaptureVideo.release()
