import cv2

cap = cv2.VideoCapture('jetson.mp4')
i=1376

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # resized_image = cv2.resize(frame, (200, 200))
        crop_img = frame[10:900, 200:1000]
        resized_image = cv2.resize(crop_img, (200, 200))
        cv2.imwrite('dataset/test/jetson/' + str(i) + '.png', resized_image)
    else:
        break

    i += 1

cap.release()