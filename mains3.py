import cv2
from datetime import datetime
import numpy as np
import getpass
import time

# ログイン名の取得
user = getpass.getuser()


cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する
net = cv2.dnn.readNet(f'/Users/{user}/Desktop/porous/text/best1.onnx')
path = f"/Users/{user}/Desktop/porous/text/a.png"
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

while True:
    t1 = time.time()  #開始の時間
    
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, -1)


    



    input_image = format_yolov5(frame) # making the image square
    blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
    net.setInput(blob)
    predictions = net.forward()


    t2 = time.time()  #終了の時間

    elapsed_time1 = t2-t1
    print(f"経過時間1：{elapsed_time1}")
    """"""

    # step 3 - unwrap the predictions to get the object detections 

    """2"""
    t3 = time.time()  #開始の時間

    class_ids = []
    confidences = []
    boxes = []

    output_data = predictions[0]

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor =  image_height / 640
    
    for r in range(25200):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.3)
    #python -m pip install opencv-python==4.5.5.62
    t4 = time.time()  #終了の時間
    elapsed_time2 = t4-t3
    print(f"経過時間2：{elapsed_time2}")
    """"""


    """3"""
    t5 = time.time()  #開始の時間
    result_class_ids = []#検出番号
    result_boxes = []#ボックス位置とかなんとか
    [[result_class_ids.append(class_ids[i]),result_boxes.append(boxes[i])] for i in indexes]
    


    t6 = time.time()  #終了の時間
    elapsed_time3 = t6-t5
    print(f"経過時間3：{elapsed_time3}")
    """"""

    """4"""
    t7 = time.time()  #開始の時間
    [cv2.rectangle(frame, result_boxes[i], (0, 255, 25), 5) for i in range(len(result_class_ids))]
    t8 = time.time()  #終了の時間
    elapsed_time4 = t8-t7
    print(f"経過時間4：{elapsed_time4}")
    """"""


    """5"""
    t9 = time.time()  #開始の時間
    cv2.putText(frame, f"Porous {len(result_class_ids)}" , (20, 160), 0, 2.5, (255, 0, 255), 3)
    date = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
    cv2.putText(frame, f"{date}" , (20, 80), 0, 1, (255, 0, 255), 3)
    
    t10 = time.time()  #終了の時間
    elapsed_time5 = t10-t9
    print(f"経過時間5：{elapsed_time5}")
    """"""
    print(f"ループにかかる時間：{elapsed_time1 + elapsed_time2 + elapsed_time3 + elapsed_time4 + elapsed_time5}")
    print(f"検出{len(result_class_ids)}")
    
    cv2.imshow("camera", frame)
    

    
    
    k = cv2.waitKey(1)&0xff # キー入力を待つ
    
        


        


    
        
    if k == ord('e'):
        # 「q」キーが押されたら終了する
        break




# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
