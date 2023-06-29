import cv2
from super_image.utils.metrics import ssim

# скрипт для вычисления ssim и psnr
img1 = cv2.imread("images/hr.png")
img2 = cv2.imread("images/real.png")

ssim_value = ssim(img1, img2)
print(ssim_value)
psnr_value = cv2.PSNR(img1, img2)
print(psnr_value)




