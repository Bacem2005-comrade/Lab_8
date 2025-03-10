import cv2
import cv2.aruco as aruco
import numpy as np
import datetime
import pygame

# Initialize Pygame for playing audio
pygame.mixer.init()
pygame.mixer.music.load('jungle.mp3')  # Replace with your MP4 file path
pygame.mixer.music.play(-1)  # Play the music indefinitely

# Image Preprocessing

# Load the elephant image
image_path = "variant-2.png"
try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    print(f"Error loading elephant image: {e}")
    exit()

# Load the fly image
fly_image_path = "fly64.png"
try:
    fly_img = cv2.imread(fly_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
    if fly_img is None:
        raise FileNotFoundError(f"Fly image not found at {fly_image_path}")
except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    print(f"Error loading fly image: {e}")
    exit()
fly_height, fly_width, fly_channels = fly_img.shape

# Function to overlay image with transparency
def overlay_image(background, overlay, x, y):
    b_height, b_width, b_channels = background.shape
    o_height, o_width, o_channels = overlay.shape
    y1, y2 = max(0, y), min(b_height, y + o_height)
    x1, x2 = max(0, x), min(b_width, x + o_width)
    overlay_y1, overlay_y2 = max(0, -y), o_height - max(0, y + o_height - b_height)
    overlay_x1, overlay_x2 = max(0, -x), o_width - max(0, x + o_width - b_width)
    roi = background[y1:y2, x1:x2]
    overlay_roi = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    if o_channels == 4:  # Assuming RGBA
        alpha = overlay_roi[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        roi[:] = alpha * overlay_roi[:, :, :3] + (1 - alpha) * roi
    else:
        roi[:] = overlay_roi

    return background

# Overlay Fly on the image
elephant_height, elephant_width, elephant_channels = img.shape
fly_x = elephant_width // 2 - fly_width // 2
fly_y = elephant_height // 2 - fly_height // 2
img = overlay_image(img, fly_img, fly_x, fly_y)
cv2.imwrite("elephant_with_fly.png", img)

# Apply Gaussian blur
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
blurred_png_image_path = "blurred_image.png"
cv2.imwrite(blurred_png_image_path, blurred_img)

# Camera Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Aruco Marker Setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# File setup for logging marker coordinates
output_file = "marker_coordinates.txt"
try:
    with open(output_file, "w") as f:
        f.write("Aruco Marker Coordinates Log\n")
        f.write(f"Session started at: {datetime.datetime.now()}\n")
        f.write("--------------------------------------\n")
except Exception as e:
    print(f"Error opening/creating file: {e}")
    exit()

# Aruco Marker Detection Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Overlay the blurred elephant image
    b_height, b_width, b_channels = blurred_img.shape
    f_height, f_width, f_channels = frame.shape
    resized_blurred = cv2.resize(blurred_img, (f_width, f_height))
    frame = resized_blurred

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i, corners_i in enumerate(corners):
            marker_center_x = int(np.mean([corner[0][0] for corner in corners_i]))
            marker_center_y = int(np.mean([corner[0][1] for corner in corners_i]))
            timestamp = datetime.datetime.now()
            log_entry = f"{timestamp}: Marker ID {ids[i][0]} Center: ({marker_center_x}, {marker_center_y})\n"
            try:
                with open(output_file, "a") as f:
                    f.write(log_entry)
                print(log_entry.strip())
            except Exception as e:
                print(f"Error writing to file: {e}")

    # Display
    cv2.imshow("Frame", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()  # Stop the music once the program ends

print(f"Marker coordinate logging finished. See {output_file}")

