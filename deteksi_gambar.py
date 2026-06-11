import cv2
import imutils

# Inisialisasi detektor HOG [cite: 27, 28]
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membaca Gambar [cite: 32]
image = cv2.imread('img.png')

# Mengubah ukuran gambar agar proses lebih cepat [cite: 34, 37]
image = imutils.resize(image, width=min(400, image.shape[1]))

# Mendeteksi pejalan kaki [cite: 40]
(regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

# Menggambar kotak di sekeliling pejalan kaki [cite: 42, 43, 44, 45]
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Menampilkan hasil [cite: 46, 47, 48, 49]
cv2.imshow("Hasil Deteksi", image)
cv2.waitKey(0)
cv2.destroyAllWindows()