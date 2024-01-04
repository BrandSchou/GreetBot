import cv2 # Importer OpenCV-biblioteket
import os # Importer os-biblioteket for at arbejde med stioperationer
import tkinter as tk


bkgr = '#009000' #set color(background)
txtc = 'black' #set txt color


## Brugt til lyd:
from pygame import mixer
import random
## -------------

mp3_files = ["UhOh.mp3", "StopShooting.mp3", "WhoAreYou.mp3", "haihaihai.mp3", "yayaya-hai.mp3"]
associateText = ["Uh oh.", "Heey! Its me! Dont shoot!", "Who are you?", "Hey hey hey!", "yayayayayayayayaya-.. Hi! :)"]

mixer.init()


precision = 60 # Hvår sikker skal programmet være at den er rigtig i %

# thres = 0.45 # Threshold to detect object
className = "person" # Klassenavn for objektet, der skal genkendes
print(className)

# Stier til konfigurations- og vægtfiler for det trænede objektdetekteringsmodel
configPath = "PY-CAM\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"#os.path.join("PY-CAM","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt") #"/home/mathiasschou/Desktop/PY-CAM/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "PY-CAM\\frozen_inference_graph.pb"#os.path.join("PY-CAM","frozen_inference_graph.pb") #"/home/mathiasschou/Desktop/PY-CAM/frozen_inference_graph.pb"

# Opret et objekt af typen cv2.dnn_DetectionModel med de angivne vægt- og konfigurationsfiler
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320) # Sæt inputstørrelsen for billedet, som modellen forventer
net.setInputScale(1.0/ 127.5)# Skalér inputbillederne ved at dividere pixelværdierne med denne faktor
net.setInputMean((127.5, 127.5, 127.5))# Specificer gennemsnitsværdier for RGB-kanalerne for billederne
net.setInputSwapRB(True)# Byt om på R og B kanaler i billederne

# --- functions ---
# Funktion til at få information om genkendte objekter i et billede
def getObjects(img, thres, nms, draw=True, objects=[]):
    checksum = False
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)

    # Hvis listen objects er tom, brug className
    if len(objects) == 0: 
        objects = className
    objectInfo =[]

    
    # Hvis der er genkendte objekter i billedet
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            print(confidence)
            con = round(confidence*100,2)
            print(con)
            
            # Hvis tillidsværdien er over 60%, udfør handlinger
            if con>precision:
                checksum = True
                if className in objects:
                    objectInfo.append([box,className])

                    #H vis draw er sandt, tegn rektangel og tekst på billedet
                    if (draw):
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2) # Tegn et rektangel omkring det genkendte objekt
                        cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),# Indsæt tekst med det genkendte klassenavn
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cv2.putText(img,str(con)+"%",(box[0]+200,box[1]+30),# Indsæt tekst med tillidsværdien
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    return img, objectInfo, checksum


def main(greet):
    # Åbn webcam
    cap = cv2.VideoCapture(0)# Opret et VideoCapture-objekt og forbind det til den første videoenhed
    cap.set(3,640)# Sæt bredden på billedet, der indfanges af kameraet, til 640 pixels
    cap.set(4,480)# Sæt højden på billedet, der indfanges af kameraet, til 480 pixels
    # cap.set(10,70)

    previousText = 69
    tim=99

    while True:
        # Læs et billede fra webcam
        success, img = cap.read()
        # Kald funktionen for at få information om genkendte objekter i billedet
        result, objectInfo, checksum = getObjects(img,0.45,0.2)
        if checksum == True:
            tim+=1 # using tim in order to not have new messeges constanly
            if tim>80: # increase number if new messege is comming to fast(remember adjusting else tim addition if altert)
                tim=0
                random_selection = random.randint(0, len(mp3_files) - 1) # Kigger på listen 'mp3_files' og vælger en random fil fra listen.
                if random_selection == previousText:
                    random_selection = random.randint(0,len(mp3_files) - 1)
                mixer.music.load(os.path.join("assets", mp3_files[random_selection])) # Her bruges pygame til at tage den random fil fra før og afspiller lyden.
                mixer.music.play()
                previousText = random_selection
                greet.config(text = associateText[random_selection]) # Her konfigureres teksten som matcher lyd filen til at blive vist på skærmen.
                root.update()
        else:
            greet.config(text = "") # Her sættes teksten tilbage til ingenting hvis checksum'en ikke er '= True'
            tim+=10 # bigger number if it takes to long to greet new person after contact was brocken

        # Vis billedet i et vindue
        cv2.imshow("Output", img)

        root.update()
        checksum = False

# --- main window ---

root = tk.Tk()
root.title("Greeting window")
root.attributes("-fullscreen", True)
root.tk_setPalette(bkgr) #sets all colors to bkgr color

# add frame in main window (root)

logo = tk.Frame(root)
logo.pack()
txt = tk.Frame(root)
txt.pack()

# build canvas

canvas=tk.Canvas(logo,width=800,height=242)
canvas.pack()
img = tk.PhotoImage(file=(os.path.join("assets","Seluxit_logo_updated_color_800px.png")))
canvas.create_image(0,0,anchor=tk.NW,image=img)
canvas.grid(row=50,column=50,pady=20,padx=100)

# put widgets in frame (txt)
greet = tk.Label(txt,text="",font=('times', 20), bd=1, anchor=tk.W,foreground=txtc)
greet.grid(row=10, column=10)

# put widget directly in main widnow (root)
tk.Button(root, text='CLOSE',foreground=txtc, command=root.destroy).pack(side= tk.RIGHT)

main(greet)

# --- start ---

root.mainloop() #needed for tkinter to work and evrything after doas not work
