import cv2
import glob


def save_images_from_video(video_path, save_dir, i=0):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            crop_img = frame[10:900, 200:1000]
            resized_image = cv2.resize(crop_img, (200, 200))
            cv2.imwrite(save_dir + str(i) + '.png', resized_image)
        else:
            break

        i += 1

    cap.release()


def resize_images_in_folder(folder_path, size=(200.200)):
    filenames = glob.glob(folder_path + "/*.png")
    filenames.sort()

    for i, img in enumerate(filenames):
        resized_image = cv2.resize(cv2.imread(img), size, interpolation=cv2.INTER_AREA)
        print(i)
        cv2.imwrite(folder_path + '/' + str(i) + '.png', resized_image)