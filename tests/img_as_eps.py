import os
from PIL import Image

if __name__ == '__main__':
    path = r"D:\Projects\pyprog\SpeechTools\files\mmm_example_default.png"
    fname = os.path.splitext(os.path.basename(path))[0] + '.eps'
    save_path = r"./files/" + fname
    img = Image.open(path, 'r')
    # 转换图像模式为RGB
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    img.save(save_path,"EPS")