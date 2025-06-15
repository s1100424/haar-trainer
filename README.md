# haar-trainer
#此專題有兩個主程式與4個副程式 以下為使用方式介紹  
1. 先準備好已經標記好想訓練物件的座標檔案(如使用makesense.ai標記後產生的包含xml檔案的資料夾)  
2.打開1.main_transfer.py 並設定好你的正樣本與負樣本資料夾路徑與名稱  
3.決定你要擴增負樣本到多少張  
4.修改trainner.create_vec_file與 trainner.train_cascade_classifier中函式的變數以確保符合你得要求  
5.如果以需要更改更多相關變數請打開trainner.py 那邊有更詳細的設定 包括記憶體 訓練型式等等  
