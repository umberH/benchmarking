# Upscale the provided composite to 7200×7200 (300-dpi print size) with gentle sharpening,
# and export PNG, TIFF, and PDF.

from PIL import Image, ImageFilter
from pathlib import Path
from fpdf import FPDF

src = Path("C://Users//dumb_//Downloads//moon_mission_custom_choice_TL-3B_TR-3A_BL-1D_BR-2D.png")
assert src.exists(), f"Source not found: {src}"
im = Image.open(src).convert("RGB")

# Target for 60×60 cm at 300 DPI
target_size = (7200, 7200)

# Resize with high-quality filter
up = im.resize(target_size, Image.Resampling.LANCZOS)

# Gentle print sharpening
up = up.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))

# Save formats
png_out  = Path("C:/Users/dumb_/Downloads/moon_mission_7200_print.png")
tif_out  = Path("C:/Users/dumb_/Downloads/moon_mission_7200_print.tif")
jpg_out  = Path("C:/Users/dumb_/Downloads/moon_mission_7200_print.jpg")
pdf_out  = Path("C:/Users/dumb_/Downloads/moon_mission_7200_print.pdf")

up.save(png_out, dpi=(300,300), optimize=True)
up.save(jpg_out, "JPEG", quality=96, optimize=True, progressive=True, dpi=(300,300))
up.save(tif_out, compression="tiff_lzw", dpi=(300,300))

# PDF export using JPEG inside for size efficiency
tmp_jpg = Path("C:/Users/dumb_/Downloads/_tmp_print.jpg")
up.save(tmp_jpg, "JPEG", quality=95, optimize=True)
pdf = FPDF(unit="pt", format=[target_size[0], target_size[1]])
pdf.add_page()
pdf.image(str(tmp_jpg), x=0, y=0, w=target_size[0], h=target_size[1])
pdf.output(str(pdf_out))

# Display a small preview
# display(up.resize((1200,1200)))
up.resize((1200,1200)).save('resized_image.png')
str(png_out), str(jpg_out), str(tif_out), str(pdf_out), up.size
