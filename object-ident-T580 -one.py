import cv2 # Importer OpenCV-biblioteket
import os # Importer os-biblioteket for at arbejde med stioperationer

precision = 60 #Hvår sikker skal programmet være at den er rigtig i %

#thres = 0.45 # Threshold to detect object


className = "person" # Klassenavn for objektet, der skal genkendes
print(className)

#Stier til konfigurations- og vægtfiler for det trænede objektdetekteringsmodel
configPath = "PY-CAM\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"#os.path.join("PY-CAM","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt") #"/home/mathiasschou/Desktop/PY-CAM/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "PY-CAM\\frozen_inference_graph.pb"#os.path.join("PY-CAM","frozen_inference_graph.pb") #"/home/mathiasschou/Desktop/PY-CAM/frozen_inference_graph.pb"

#Opret et objekt af typen cv2.dnn_DetectionModel med de angivne vægt- og konfigurationsfiler
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320) # Sæt inputstørrelsen for billedet, som modellen forventer
net.setInputScale(1.0/ 127.5)# Skalér inputbillederne ved at dividere pixelværdierne med denne faktor
net.setInputMean((127.5, 127.5, 127.5))# Specificer gennemsnitsværdier for RGB-kanalerne for billederne
net.setInputSwapRB(True)# Byt om på R og B kanaler i billederne


#Funktion til at få information om genkendte objekter i et billede
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)

    #Hvis listen objects er tom, brug className
    if len(objects) == 0: 
        objects = className
    objectInfo =[]

    
    #Hvis der er genkendte objekter i billedet
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            print(confidence)
            con = round(confidence*100,2)
            print(con)
            
            #Hvis tillidsværdien er over 60%, udfør handlinger
            if con>precision:
                if className in objects:
                    objectInfo.append([box,className])

                    #Hvis draw er sandt, tegn rektangel og tekst på billedet
                    if (draw):
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2) # Tegn et rektangel omkring det genkendte objekt
                        cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),# Indsæt tekst med det genkendte klassenavn
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cv2.putText(img,str(con)+"%",(box[0]+200,box[1]+30),# Indsæt tekst med tillidsværdien
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo

# Hovede programmet starter her. 
if __name__ == "__main__":
    #Åbn webcam
    cap = cv2.VideoCapture(0)# Opret et VideoCapture-objekt og forbind det til den første videoenhed
    cap.set(3,640)# Sæt bredden på billedet, der indfanges af kameraet, til 640 pixels
    cap.set(4,480)# Sæt højden på billedet, der indfanges af kameraet, til 480 pixels
    #cap.set(10,70)


    while True:
        #Læs et billede fra webcam
        success, img = cap.read()
        # Kald funktionen for at få information om genkendte objekter i billedet
        result, objectInfo = getObjects(img,0.45,0.2)
        
        #Vis billedet i et vindue
        cv2.imshow("Output",img)

        #Vent på et tastetryk i 1 millisekund
        cv2.waitKey(1)
