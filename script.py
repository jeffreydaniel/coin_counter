from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('model.pt') # Change model if necessary

# Define a dictionary to map class labels to their corresponding coin values
coin_values = {
    "1_rupeescoin": 1,
    "2_rupeescoin": 2,
    "5_rupeescoin": 5,
    "10_rupeescoin": 10,
    "20_rupeescoin": 20
}

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Run inference on the frame
        results = model(frame, stream=True)

        # Initialize variables to store the total number of coins and the total sum
        total_coins = 0
        total_sum = 0

        # Loop through the detected objects
        for result in results:
            detection = result.boxes

            for box in detection:
                class_id = box.cls[0].tolist()  # Get the scalar value from the tensor
                class_name = model.names[int(class_id)]

                # Check if the class name is in the coin_values dictionary
                if class_name in coin_values:
                    total_coins += 1
                    total_sum += coin_values[class_name]

                    # Draw the bounding box and label on the frame
                    x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # Display the total number of coins and the total sum on the frame
        cv2.putText(frame, f"No_of_coins: {total_coins}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Sum_of_coins: {total_sum}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the output frame
        cv2.imshow('Output', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
