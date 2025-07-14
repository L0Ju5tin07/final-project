import jetson.inference
import jetson.utils
import cv2

net = jetson.inference.detectNet("pednet", threshold=0.5)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cuda_img = jetson.utils.cudaFromNumpy(img_rgba)

    detections = net.Detect(cuda_img)

    for detection in detections:
        if net.GetClassDesc(detection.ClassID) == "person":
            left = int(detection.Left)
            top = int(detection.Top)
            right = int(detection.Right)
            bottom = int(detection.Bottom)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("People Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()