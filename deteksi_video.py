import cv2
import imutils

# Inisialisasi detektor HOG [cite: 55, 56, 57]
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membaca video [cite: 59]
cap = cv2.VideoCapture('vid.mp4')

while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))
        
        # Mendeteksi pejalan kaki [cite: 71, 72, 73, 74]
        (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
        
        # Menggambar kotak [cite: 75, 77, 78, 79, 80]
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
        cv2.imshow("Deteksi Video", image)
        
        # Tekan 'q' untuk berhenti [cite: 83, 84]
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()