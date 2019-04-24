import cv2
import glob

filenames = glob.glob("dataset_main/synthetic/rgb/jetson/*.png")
filenames.sort()

for i, img in enumerate(filenames):
    resized_image = cv2.resize(cv2.imread(img), (200, 200), interpolation=cv2.INTER_AREA)
    print(i)
    cv2.imwrite('dataset_main/synthetic/rgb/jetson/' + str(i) + '.png', resized_image)