import cv2
import numpy as np

def merge_boxes(boxes, iou_threshold=0.5):
    from imutils.object_detection import non_max_suppression
    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])
    pick = non_max_suppression(rects, probs=None, overlapThresh=iou_threshold)
    return [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in pick]

# 請先建立一個名為 camera_config.txt 的檔案，內容格式如下：
#網路攝影機帳號
#網路攝影機密碼
#網路攝影機IP
with open("camera_config.txt", "r") as f:
    lines = f.read().splitlines()

user = lines[0]
password = lines[1]
ip = lines[2]
video = f"http://{user}:{password}@{ip}/video"
print(f"Connecting to video stream: {video}")
cap = cv2.VideoCapture(video)# 開啟網路攝影機

# 3 400 600
#參數設置
minS = 100
maxS = 400
# 請將以下路徑替換為您的分類器檔案路徑
# 注意：這個分類器應該是您之前訓練好的分類器
pictPath = r'C:\Users\admin\source\repos\opencv\fin_test\classifier\cascade.xml' 
face_cascade = cv2.CascadeClassifier(pictPath) # 建立辨識物件

#攝像頭抓取
while True:
    ret, img = cap.read() # 讀取影像
    #cv2.resize(img, (360, 240)) # 調整影像大小
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)
    if not cap.isOpened():
        print("Error: Cannot open video stream")
        exit()
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2,minNeighbors=4, minSize=(minS, minS), maxSize=(maxS, maxS))
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
