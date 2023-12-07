import cv2
import os

#thres = 0.45 # Threshold to detect object


className = "person"
print(className)

configPath = "C:\\Users\\n\\Documents\\Dania\\Gruppe_5\\Projekt\\GreetBot-main\\GreetBot-main\\PY-CAM\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"#os.path.join("PY-CAM","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt") #"/home/mathiasschou/Desktop/PY-CAM/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "C:\\Users\\n\\Documents\\Dania\\Gruppe_5\\Projekt\\GreetBot-main\\GreetBot-main\\PY-CAM\\frozen_inference_graph.pb"#os.path.join("PY-CAM","frozen_inference_graph.pb") #"/home/mathiasschou/Desktop/PY-CAM/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = className
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            print(confidence)
            con = round(confidence*100,2)
            print(con)
            if con>60:
                if className in objects:
                    objectInfo.append([box,className])
                    if (draw):
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                        cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cv2.putText(img,str(con),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)


    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2)
        #print(objectInfo)
        cv2.imshow("Output",img)
        cv2.waitKey(1)
