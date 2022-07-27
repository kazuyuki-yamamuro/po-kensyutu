# 必要モジュールをインポートします
import cv2
from datetime import datetime
import numpy as np
import getpass



'''使い回すメソッド、関数、パスを定義'''

# どのPCでもパスを通せるようにログイン名の取得
user = getpass.getuser()

# 任意のカメラ番号を指定してビデオキャプチャーを変数化
cap = cv2.VideoCapture(0)

# フォルダに絶対パスを通してonnxファイルを読み込む
net = cv2.dnn.readNet(f'/Users/{user}/Desktop/porous/text/best1.onnx')

# フォルダに絶対パスを通して検出画像を一時的に保存するのに使う
path = f"/Users/{user}/Desktop/porous/text/a.png"

# 読み込んだ画像を正方形にする関数を用意しておきます
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result





# 以下のwhile文をループ中に処理した画像を見ることがこのアプリケーションの活用方法になります

while True:
    ret, frame = cap.read() # カメラから画像を取得する
    
    frame = cv2.flip(frame, -1) # 取得した画像の上下をひっくり返すことで人が見ている景色とPC画面を同じにする


    # キャプチャーした画像にキー入力説明を入れてアプリの使い方を常に表示する
    cv2.putText(frame, "C:camera" , (20, 200), 2, 1, (150, 150, 50), 2)
    cv2.putText(frame, "D:detection", (20, 240), 2, 1, (150, 150, 50), 2)
    cv2.putText(frame, "E:exit" , (20, 350), 2, 1, (150, 150, 50), 2)


    cv2.imshow("camera", frame) # キャプチャーした画像を表示する
    
    k = cv2.waitKey(10)&0xff # キー入力を待つ入力によって処理を分岐する

    if k == ord('d'): # 「d」キーが押されたら以下の処理を行う
        

        # キャプチャーした画像をyolov5で作られたONNXファイルで検出できるように加工
        # yolov5から作ったONNXファイルの特性上検出に使えるのは640✖️640pxの画像になるので変換
        # 1/255の倍率を使用して、画像のピクセル値を0から1のターゲット範囲にスケーリングします
        # 加工後の画像で予測行う
        input_image = format_yolov5(frame) # making the image square
        blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
        net.setInput(blob)
        predictions = net.forward()



        # 予測をアンラップしてオブジェクトの検出を取得します
        # yolov5で作られたAIは640✖️640の画像内に25200のバウンディングボックスが存在するので全部検出をループします。
        # 良好な検出を除外します
        # 最高のクラススコアのインデックスを取得します。
        # クラススコアがしきい値より低い検出は破棄します。
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
            if confidence >= 0.4: # 0.4はSCORE_THRESHOLDでその値以下の自信を含んでいる任意のバウンディングボックスを排除します
                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25): # .25が閾値でこの値より下の境界ボックスを削除する

                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    #　見やすいように少しだけ獲得座標をずらします
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)


        # opencv-python==4.5.5.62じゃないとエラーになる
        # boxes, confidencesをarray型からcv2で座標出力できる形に成形してindexesリストに入れる
        # 0.25が閾値で0.4はSCORE_THRESHOLDです基本array型を作ったときと同じ値にします
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.4)
        


        
        
        result_class_ids = [] # 検出番号を格納するリストを作成&初期化
        result_boxes = [] # ボックス位置座標を格納するリストを作成&初期化

        # 篩にかけてindexesリストになったデータを使ってループし閾値以上かつ重複対策後の検出番号とボックス位置座標をリストに入れる
        [[result_class_ids.append(class_ids[i]),result_boxes.append(boxes[i])] for i in indexes]
        
        # ボックス位置座標を検出番号に対してlenを使って検出数とする。検出数分ループすることで検出された座標を獲得しそのまま座標にcv2で四角を描写する
        [cv2.rectangle(frame, result_boxes[i], (0, 255, 25), 5) for i in range(len(result_class_ids))]
        


        
        
        # 今この瞬間の時間を秒単位まで取得して文字列でdate変数に入れる
        date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        # date変数にいれた時間文字列を画像に書き込む
        cv2.putText(frame, f"{date}" , (20, 80), 0, 1, (255, 0, 255), 3)

        # 物体検出した数を画像に書き込む
        cv2.putText(frame, f"Porous {len(result_class_ids)}" , (20, 160), 0, 2.5, (255, 0, 255), 3)

        cv2.imwrite(path, frame) # pathの階層にa.pngという名前で画像を保存する
        cv2.imshow(path, frame) # 保存した画像を表示する
        
        
        




    elif k == ord('c'): # 「c」キーが押されたらキャプチャした画像を表示する
        cv2.imshow(path, frame) 
  
        
    elif k == ord('e'):# 「e」キーが押されたら終了する
        break




# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
