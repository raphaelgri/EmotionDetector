from mtcnn import MTCNN
import cv2
import pprint

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
        if ret:
            a = detector.detect_faces(frame)
            if a:
                x = a[0]["box"][0]
                y = a[0]["box"][1]
                width = a[0]["box"][2]
                height = a[0]["box"][3]
                cv2.rectangle(frame, (x,y), (x+width, y+height), colour, 2)
                cv2.imshow("a", frame)
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

    # a = detector.detect_faces(cv2.imread("gr.jpg", cv2.COLOR_BGR2RGB))
    # pprint.pprint(a)
    #
    # cap.release()

    # img = cv2.cvtColor(cv2.imread("a.jpg"))
    # a = cv2.VideoCapture()
    # detector = MTCNN()
    # a = detector.detect_faces(img)
    # pprint.pprint(a)
    # print('Device Name: '+tf.test.gpu_device_name())
    #print(a)
    # x = torch.rand(5, 3)
    # x = torch.cuda.is_available()
    # print(x)