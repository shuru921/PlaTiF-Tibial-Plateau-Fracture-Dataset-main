import cv2
import matplotlib.pyplot as plt

# 讀取影像
img = cv2.imread("./yolo_dataset/images/train/Patient_ID_001.jpg")
h, w, _ = img.shape

# 讀取標註 (從你提供的內容)
class_id, x_c, y_c, wb, hb = 0, 0.447558, 0.758502, 0.340874, 0.482011

# 換算回像素座標
x1 = int((x_c - wb / 2) * w)
y1 = int((y_c - hb / 2) * h)
x2 = int((x_c + wb / 2) * w)
y2 = int((y_c + hb / 2) * h)

# 畫框並顯示
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
