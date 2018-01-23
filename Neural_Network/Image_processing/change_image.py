from PIL import Image, ImageEnhance
import cv2
import numpy as np

brightness_rate = 3.0
contrast_rate = 10.0
color_rate = 0.4
peak = Image.open("image.jpg")


# Change Brightness
enhancer = ImageEnhance.Brightness(peak)
bright = enhancer.enhance(brightness_rate)
bright.save("Brightness.jpg")

# Change contrast
contrast = ImageEnhance.Contrast(peak)
contrast = contrast.enhance(contrast_rate) # set FACTOR > 1 to enhance contrast, < 1 to decrease
contrast.save("Contrast.jpg")

#change color
color = ImageEnhance.Color(peak)
color = color.enhance(color_rate)
color.save("color.jpg")
