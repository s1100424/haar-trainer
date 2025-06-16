import cv2
import numpy as np


video = "http://<account>:<password>@<ip>/video" # 這是網路攝影機的網址
cap = cv2.VideoCapture(video)# 開啟網路攝影機
# 3 400 600
def merge_boxes(boxes, iou_threshold=0.5):
    from imutils.object_detection import non_max_suppression
    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])
    pick = non_max_suppression(rects, probs=None, overlapThresh=iou_threshold)
    return [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in pick]
#參數設置
minS = 500
maxS = 1800

pictPath = r'C:\Users\admin\source\repos\opencv\fin\classifier\cascade.xml'
face_cascade = cv2.CascadeClassifier(pictPath) # 建立辨識物件
img = cv2.imread("20250613_071945467_iOS.jpg") # 讀取影像


while True:
    ret, img = cap.read() # 讀取影像
    #print(frame)
    cv2.resize(img, (360, 240)) # 調整影像大小
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    #cv2.imshow("Camera", frame)# 顯示影像
    if not cap.isOpened():
        print("Error: Cannot open video stream")
        exit()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=5, minSize=(minS, minS), maxSize=(maxS, maxS))
    merged_faces = merge_boxes(faces)
    # 標註右下角底色是黃色
    cv2.rectangle(img, (img.shape[1], 0), (img.shape[1]-300, 30), (0, 255, 255), -1)
    # 標註找到多少的人臉
    cv2.putText(img, "Finding " + str(len(merged_faces)) + " oloo",(img.shape[1]-250, 20),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
    # 將人臉框起來, 由於有可能找到好幾個臉所以用迴圈繪出來
    for (x, y, w, h) in merged_faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 8) # 藍色框住人臉
    print(f"Finding " + str(len(merged_faces))+"個oloo")
    cv2.imshow("Face", img) # 顯示影像
    if cv2.waitKey(1) & 0xFF == ord('q'): # 按下 q 鍵就跳出迴圈
        break

cv2.destroyAllWindows()
cap.release() # 釋放攝影機

