import xml.etree.ElementTree as ET
import os

def convert_xml_to_pos_txt(xml_folder, img_folder, output_txt):
    with open(output_txt, 'w') as out_file:
        for xml_file in os.listdir(xml_folder):
            if not xml_file.endswith('.xml'):
                continue
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename = root.find('filename').text

            for obj in root.iter('object'):
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                width = xmax - xmin
                height = ymax - ymin

                img_path = os.path.join(img_folder, filename)
                out_file.write(f"{img_path} 1 {xmin} {ymin} {width} {height}\n")
    print(f"正樣本路徑資料已儲存在: {output_txt}")

def create_negative_txt(neg_folder, output_file):
    with open(output_file, 'w') as f:
        for filename in os.listdir(neg_folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                f.write(os.path.join(neg_folder, filename) + '\n')
    print(f"負樣本路徑資料已儲存在: {output_file}")