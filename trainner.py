import subprocess
import os

'''
可以在這裡修改包括crate_sample與train_cascade_classifier的參數
'''
def create_vec_file(info_txt='positive_fixed.txt', vec_output='samples.vec', num=1000, w=64, h=128):
    if not os.path.isfile(info_txt):
        print(f"[錯誤] 找不到正樣本標註檔：{info_txt}")
        return

    cmd = [
        'opencv_createsamples',
        '-info', info_txt,
        '-num', str(num),
        '-w', str(w),
        '-h', str(h),
        '-vec', vec_output
    ]
    print("[執行] 建立 .vec 檔案...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("[錯誤輸出]", result.stderr)

def train_cascade_classifier(vec_file, bg_txt, data_dir,
                             numPos=330, numNeg=1200, numStages=1, w=64, h=128):
    os.makedirs(data_dir, exist_ok=True)

    cmd = [
        'opencv_traincascade',
        '-data', data_dir,
        '-vec', vec_file,
        '-bg', bg_txt,
        '-numPos', str(numPos),
        '-numNeg', str(numNeg),
        '-numStages', str(numStages),
        '-w', str(w),
        '-h', str(h),
        '-featureType', 'LBP',
        '-mode', 'BASIC',
        '-minHitRate', '0.95',
        '-maxFalseAlarmRate', '0.3',
        '-precalcValBufSize', '1024',
        '-precalcIdxBufSize', '1024'
    ]
    print("[執行] 開始訓練分類器...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("[錯誤輸出]", result.stderr)
