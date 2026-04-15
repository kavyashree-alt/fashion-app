from rembg import remove
from PIL import Image

def segment_clothing(input_path, output_path="segmented.png"):
    input_image = Image.open(input_path).convert("RGBA")
    output = remove(input_image)

    bg = Image.new("RGB", output.size, (255, 255, 255))
    bg.paste(output, mask=output.split()[3])

    bg.save(output_path)

    return output_path