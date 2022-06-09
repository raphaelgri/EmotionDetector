from mtcnn import MTCNN
import cv2
import pprint

import image_transformer

if __name__ == '__main__':

    cap = cv2.VideoCapture('video.mp4')
    count = 0
    step = 1
    end = 300
    colour = (255, 255, 255)
    wait = 1 # 0 would lead to a crash

    detector = MTCNN()

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            a = detector.detect_faces(frame)
            if a:
                x = a[0]["box"][0]
                y = a[0]["box"][1]
                width = a[0]["box"][2]
                height = a[0]["box"][3]
                # cv2.rectangle(frame, (x,y), (x+width, y+height), colour, 2)
                croped_frame = frame[y:(y+height), x:(x+width)]
                transf = image_transformer.TransformImage()
                croped_frame = transf.resize_and_pad(croped_frame)
                croped_frame = cv2.cvtColor(croped_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("a", croped_frame)
                # cv2.imwrite("frames/output{:d}.jpg".format(count), croped_frame)
                pprint.pprint(a)
            count += step
            cv2.waitKey(wait)
            if count > end:    # Just for this test
                cap.release()
                break
        else:
            cap.release()
            break

    cv2.waitKey(wait)
    cv2.destroyAllWindows()
