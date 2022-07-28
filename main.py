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
    cv2.putText(frame, "C:camera" , (20, 200), 2, 1, (20, 50, 30), 2)
    cv2.putText(frame, "D:detection", (20, 240), 2, 1, (20, 50, 30), 2)
    cv2.putText(frame, "E:exit" , (20, 340), 2, 1, (20, 50, 30), 2)


    cv2.imshow("camera", frame) # キャプチャーした画像を表示する
    
    k = cv2.waitKey(10)&0xff # キー入力を待つ入力によって処理を分岐する

    if k == ord('d'): # 「d」キーが押されたら以下の処理を行う
        

        # キャプチャーした画像をyolov5で作られたONNXファイルで検出できるように加工
        # yolov5から作ったONNXファイルの性質上検出に使えるのは640✖️640pxの画像になるので変換
        # 1/255の倍率を使用して、画像のピクセル値を0から1のターゲット範囲にスケーリングします
        # 加工後の画像で予測行う
        input_image = format_yolov5(frame)
        blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
        net.setInput(blob)
        predictions = net.forward()



        

        confidences = [] # 検出された全ての検出精度を格納するリスト作成 & 初期化
        boxes = [] # 検出された全てのバウンディングボックス座標を格納するリスト作成 & 初期化

        # 予測をアンラップしてオブジェクトの検出を取得します
        # yolov5で作られたAIは640✖️640の画像内に25200のバウンディングボックスが存在するので全部検出をループします。
        # 良好な検出を除外します
        # 最高のクラススコアのインデックスを取得します。
        # クラススコアがしきい値より低い検出は破棄します。
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
                if (classes_scores[class_id] > .4): # .4が閾値でこの値より下の境界ボックスを削除する
                    # 精度リスト追加
                    confidences.append(confidence)
                    # バウンディングボックス座標作成
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    #　バウンディングボックスが見やすいように少しだけ獲得座標をずらします
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        
        #厳選するのが目的
        # confidencesのスコアとboxesの検出範囲から、スコアの最も良いものを軸にして検出範囲がダブっているものを排除したナンバーをindexesリストに入れる
        # 第３引数の0.4が閾値で第4引数の0.4はSCORE_THRESHOLDです
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
        
        
        

        # indexesリストに入っている厳選された番号を使って厳選されたバウンディングボックス位置座標をそのまま座標にcv2で四角を描写する
        [cv2.rectangle(frame, boxes[i], (0, 255, 25), 5) for i in indexes]



        
        
        # 今この瞬間の時間を秒単位まで取得して文字列でdate変数に入れる
        date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        # date変数にいれた時間文字列を画像に書き込む
        cv2.putText(frame, f"{date}" , (20, 80), 0, 1, (255, 0, 255), 3)

        # 物体検出した数を画像に書き込む
        cv2.putText(frame, f"Porous {len(indexes)}" , (20, 160), 0, 2.5, (255, 0, 255), 3)

        cv2.imwrite(path, frame) # pathの階層にa.pngという名前で画像を保存する
        cv2.imshow(path, frame) # 保存した画像を表示する

        print("------------------変数デバッグ開始--------------------")
        # printデバッグ用confidences, boxes
        print(f"厳選前の検出数{len(confidences)}")
        print(f"confidences:{confidences}")
        print(f"confidences変数は検出された全ての検出精度を格納するリスト")
        print()
        print(f"boxes:{boxes}")
        print(f"boxes変数は検出された全てのバウンディングボックス座標を格納するリスト")
        print()
        # printデバッグ用indexes
        print(f"indexes:{indexes}")
        print(f"indexesはconfidencesのスコアとboxesの検出範囲から、スコアの最も良いものを軸にして検出範囲がダブっているものを排除したナンバーの入ったリスト")
        print(f"厳選後の検出数:{len(indexes)}")
        print("------------------変数デバッグ終了--------------------")

        
        
        




    elif k == ord('c'): # 「c」キーが押されたらキャプチャした画像を表示する
        cv2.imshow(path, frame) 
  
        
    elif k == ord('e'):# 「e」キーが押されたら終了する
        break




# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
