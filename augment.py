import cv2
import os
import random
import numpy as np
def negative_samples(input_dir, output_dir, output_txt,target_total,rotation_range=15,
                           min_scale=0.125, max_scale=0.25):
    """
    擴增負樣本圖片，隨機裁切、旋轉，並儲存到指定資料夾。
    """
    os.makedirs(output_dir, exist_ok=True)

    source_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not source_images:
        raise ValueError("❌ 找不到任何圖片在 fin/negative/")

    samples_per_image = target_total // len(source_images) + 1
    counter = 0

    with open(output_txt, 'w') as f_out:
        for img_name in source_images:
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            for _ in range(samples_per_image):
                if counter >= target_total:
                    break

                # 隨機決定裁切尺寸
                scale = random.uniform(min_scale, max_scale)
                crop_w = int(w * scale)
                crop_h = int(h * scale)

                x = random.randint(0, w - crop_w)
                y = random.randint(0, h - crop_h)
                crop = img[y:y + crop_h, x:x + crop_w]

                # 隨機旋轉
                angle = random.uniform(-rotation_range, rotation_range)
                center = (crop.shape[1] // 2, crop.shape[0] // 2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(crop, rot_mat, (crop.shape[1], crop.shape[0]),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                outname = f"neg_{counter}.jpg"
                outpath = os.path.join(output_dir, outname)
                cv2.imwrite(outpath, rotated)

                # 寫入 txt（只寫相對路徑）
                f_out.write(outpath.replace('\\', '/') + '\n')
                counter += 1

    print(f"擴增完成，共產生 {counter} 張負樣本，儲存在 {output_dir}/")

def rotate_image_and_bbox(image, bbox, angle):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h))

    x, y, bw, bh = bbox
    points = np.array([[x, y], [x + bw, y], [x, y + bh], [x + bw, y + bh]])
    ones = np.ones((points.shape[0], 1))
    points_ones = np.hstack([points, ones])
    rotated_points = M.dot(points_ones.T).T

    rx, ry = rotated_points[:,0], rotated_points[:,1]
    x_new, y_new = int(rx.min()), int(ry.min())
    w_new, h_new = int(rx.max() - rx.min()), int(ry.max() - ry.min())
    return rotated_img, (x_new, y_new, w_new, h_new)


def flip_image_and_bbox(image, bbox):
    flipped_img = cv2.flip(image, 1)
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    x_new = w - (x + bw)
    return flipped_img, (x_new, y, bw, bh)

def positive_samples(input_txt, output_folder, output_txt):
    """ 擴增正樣本圖片，進行旋轉和翻轉，並儲存到指定資料夾。
    """
    os.makedirs(output_folder, exist_ok=True)
    with open(input_txt, 'r') as f_in, open(output_txt, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_path = parts[0]
            num_obj = int(parts[1])
            bbox = tuple(map(int, parts[2:6]))  # x y w h

            img = cv2.imread(img_path)
            if img is None:
                print(f"無法讀取圖片: {img_path}")
                continue

            basename = os.path.splitext(os.path.basename(img_path))[0]

            # 1. 原圖 + 原bbox（保留）
            new_img_path = os.path.join(output_folder, f"{basename}_orig.jpg")
            cv2.imwrite(new_img_path, img)
            f_out.write(f"{new_img_path.replace(os.sep, '/')} 1 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


            # 2. 旋轉 -15度、15度
            for angle in [-15, 15]:
                rot_img, rot_bbox = rotate_image_and_bbox(img, bbox, angle)
                new_img_path = os.path.join(output_folder, f"{basename}_rot{angle}.jpg")
                cv2.imwrite(new_img_path, rot_img)
                f_out.write(f"{new_img_path.replace(os.sep, '/')} 1 {rot_bbox[0]} {rot_bbox[1]} {rot_bbox[2]} {rot_bbox[3]}\n")

            # 3. 水平翻轉
            flip_img, flip_bbox = flip_image_and_bbox(img, bbox)
            new_img_path = os.path.join(output_folder, f"{basename}_flip.jpg")
            cv2.imwrite(new_img_path, flip_img)
            f_out.write(f"{new_img_path.replace(os.sep, '/')} 1 {flip_bbox[0]} {flip_bbox[1]} {flip_bbox[2]} {flip_bbox[3]}\n")
    print(f"正樣本擴增完成！儲存在 {output_folder}/")