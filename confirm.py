import os
import cv2

def clean_positive_file(input_txt,  output_txt, image_root = '',):
    with open(input_txt, 'r') as infile:
        lines = infile.readlines()

    clean_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            print(f"[跳過] 格式錯誤：{line.strip()}")
            continue

        img_path_original = parts[0]
        #img_path_no_prefix = img_path_original.replace("fin/", "").replace("fin\\", "")
        full_path = os.path.normpath(os.path.join(image_root, img_path_original))


        if not os.path.isfile(full_path):
            print(f"[錯誤] 找不到圖片：{full_path}")
            continue

        try:
            img = cv2.imread(full_path)
            if img is None:
                print(f"[錯誤] 無法讀取圖片：{full_path}")
                continue

            h, w = img.shape[:2]
            x = int(parts[2])
            y = int(parts[3])
            box_w = int(parts[4])
            box_h = int(parts[5])

            if x < 0 or y < 0 or x + box_w > w or y + box_h > h:
                print(f"[跳過] 框超出邊界：{img_path_original}, 框=({x},{y},{box_w},{box_h}), 圖片=({w},{h})")
                continue

            # 不改變原本圖片路徑寫法
            clean_lines.append(f"{img_path_original} 1 {x} {y} {box_w} {box_h}\n")

        except Exception as e:
            print(f"[錯誤] {full_path} 發生例外：{e}")

    with open(output_txt, 'w') as out:
        out.writelines(clean_lines)

    print(f"\n✅ 修正完成，共留下 {len(clean_lines)} 筆合法資料，已存至：{output_txt}")