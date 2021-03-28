import cv2
import numpy 
import imutils
import time
import tensorflow
import tensornets


# vehicles = ["car","bike","bus","truck", "auto rickshaw"]
# cv2.putText(frame, 'Total Vehiles : {}'.format(vehile), (450, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)


# # cv2.line(f, (x1, y1), (x2, y2), (0,0,255), 3) #red line
# # cv2.line(f, (x1, y1-10), (x2, y2-10), (255,0,0), 1.5) #blue top offset line
# # cv2.line(f, (x1, y1+10), (x2, y2+10), (255,0,0), 1.5) #blue bottom offset line

# cv2.line(f, (0, 450), (1500, 450), (0,0,255), 3) #red line
# cv2.line(f, (0, 440), (1500, 440), (255,0,0), 1) #blue top offset line
# cv2.line(f, (0, 460), (1500, 460), (255,0,0), 1) #blue bottom offset line



intake = tensorflow.placeholder(tensorflow.float32, [None, 416, 416, 3])
mask_nets = nets.YOLOv3COCO(intake, nets.Darknet19)

classes_dict = {'1':'rickshaw','2':'car','3':'bike','5':'bus','6':'truck'}
classes_arr = [1,2,3,5,6]

with tensorflow.Session() as sesh:
    sesh.run(mask_nets.pretrained())
    CaptureVideo = cv2.VideoCapture(r"/home/aditya/Desktop/ML+CV/1615363610851.mp4") #reading video
    while (CaptureVideo.isOpened()):
        ret, frame = CaptureVideo.read()
        image = cv2.resize(frame,(416,416))
        img_array = numpy.array(image).reshape(-1,416,416,3)
        commencement_time = time.time()
        predictions = sesh.run(tf.keras.Model.predict(mask_nets), {intake: mask_nets.preprocess(img_array)})
        contours = mask_nets.get_boxes(predictions, image.shape[1:3])
        cv2.namedWindow("IMAGE", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("IMAGE", 700,700)
        contours_array = numpy.array(contours)
        for i in classes_arr:
            count = 0
            if str(i) in classes_dict:
                vehicle_type = classes_dict[str(i)]
            if(len(contours_array==0)):
                break
            else:
                j = 0
                while(j<len(contours_array[i])):
                    cntr = contours_array[i][j]
                    if (contours_array[i][j][4]>=0.40):
                        count+=1
                        cv2.rectangle(image,(contours_array[0],contours_array[1]),(contours_array[2],contours_array[3]),(0,255,0),1)
                        cv2.putText(frame, 'Total Vehiles : {}'.format(count), (450, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        cv2.imshow("IMAGE", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cv2.destroyAllWindows()
CaptureVideo.release()
