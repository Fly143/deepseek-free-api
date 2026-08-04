"""用 DeepSeek 官方图标生成全套 launcher 密度。"""
from PIL import Image
import os

SRC = "D:/DeepSeek-Android/ds_icon.png"
OUT = "D:/DeepSeek-Android/app/src/main/res"
SIZES = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}

src = Image.open(SRC).convert("RGBA")

for name, px in SIZES.items():
    folder = os.path.join(OUT, "mipmap-" + name)
    os.makedirs(folder, exist_ok=True)
    icon = src.resize((px, px), Image.LANCZOS)
    icon.save(os.path.join(folder, "ic_launcher.png"))
    icon.save(os.path.join(folder, "ic_launcher_round.png"))
    print(f"{name}: {px}x{px} OK")

src.resize((512, 512), Image.LANCZOS).save("D:/DeepSeek-Android/icon_preview.png")
print("preview OK")
