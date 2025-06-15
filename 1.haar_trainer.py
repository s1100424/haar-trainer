import os 
import conver_txt
import augment
import confirm
import trainner

if __name__ == '__main__':
    # 轉換xml標記檔為正樣本txt檔
    # 注意：annotations_xml_folder裡的資料應該以事先透過其他工具如makesense.ai標記
    xml_folder = 'annotations_xml_folder'    #標記完成的xml資料夾位置
    pos_folder = 'positive'    #正樣本資料夾位置
    pos_txt = 'positives.txt'  #正樣本txt命名與位置
    neg_folder = 'negative'    #負樣本資料夾位置
    neg_txt = 'negatives.txt'  #負樣本txt命名與位置
    conver_txt.convert_xml_to_pos_txt(xml_folder, pos_folder, pos_txt)
    #conver_txt.create_negative_txt(neg_folder, neg_txt)

    #透過旋轉,裁切,縮放等步驟, 擴增正 負樣本數量
    ng_aug_dir = 'ng_augment'    #負樣本擴增後資料夾位置
    ng_aug_txt = 'ng_augment.txt'    #負樣本擴增後txt命名與位置

    pos_aug_dir = 'pos_augment'    #正樣本擴增後資料夾位置
    pos_aug_txt = 'pos_augment.txt'    #正樣本擴增後txt命名與位置

    target_total = 1500    #負樣本目標擴增數量
    # rotation_range (最大旋轉角度)預設15,min_scale(最小裁切比例)預設0.125,max_scale(最大裁切比例)預設0.25
    augment.negative_samples(neg_folder, ng_aug_dir, ng_aug_txt,target_total) 
    augment.positive_samples(pos_txt, pos_aug_dir, pos_aug_txt)

    #剔除正樣本txt中的錯誤資料
    output_txt='positive_fixed.txt'
    confirm.clean_positive_file(pos_aug_txt, output_txt)

    #建立vec
    trainner.create_vec_file(info_txt=output_txt, vec_output='samples.vec', num=349, w=64, h=128)

    #訓練分類器
    '''
    這邊建議用cmd指令來訓練分類器, 這樣可以看到詳細的訓練過程與錯誤訊息
    直接用python跑沒有辦法實時的看到訓練狀況 比方 N |HR | FA 還有第幾個SATGE等
    '''
    trainner.train_cascade_classifier(vec_file='samples.vec', bg_txt=ng_aug_txt, data_dir='classifier',numPos=310, numNeg=1200, numStages=6, w=64, h=128)
    #慎用!!!!
'''開始訓練
1. 建立''資料夾
mkdir classifier
打改cmd 
1. conda activate opencv
2. cd <您的專案資料夾路徑>
3.開始訓練 訓練參數:
opencv_traincascade -data classifier -vec samples.vec -bg ng_augment.txt -numPos 330 -numNeg 1200 -numStages 6 -w 64 -h 128 -featureType LBP -mode BASIC 
-minHitRate 0.95 -maxFalseAlarmRate 0.3 -precalcValBufSize 1024 -precalcIdxBufSize 1024
'''