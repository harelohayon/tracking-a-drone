import cv2
import numpy as np
import matplotlib.pyplot as plt
from classUKF2D import UKF2D

#Center settings
cap = cv2.VideoCapture(r"C:\Users\HARIEL\OneDrive\Desktop\projectUFK\ufkxy\nolinear.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
dt  = 1000 / fps  
delay_ms = int(round(1000.0 / fps))
ROTATE = cv2.ROTATE_90_CLOCKWISE  
BOX = 6 
measurements = [] 
estimates    = []

# Kalman's first variables
x0 = np.zeros(4)
P0 = np.diag([100, 100, 1000, 1000])
Pv = np.diag([0.001, 0.001])   
Pn = np.diag([4, 4])          
ukf = UKF2D(dt, x0, P0, Pv, Pn)

# Detection Function
def find_led(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (15, 15), 0) 
    _, _, _, maxLoc = cv2.minMaxLoc(g)  
    return maxLoc

# Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, ROTATE)
    cameraPoint = find_led(frame)
    ukf.predict() 
    ukf.update([cameraPoint[0], cameraPoint[1]]) 
    measurements.append(cameraPoint)
    estimates.append((ukf.x[0], ukf.x[1]))

    # Drawing 
    # Blue box - Camera 
    zx, zy = cameraPoint[0], cameraPoint[1] # Extract coordinates for drawing
    cv2.rectangle(frame, (zx-2*BOX, zy-2*BOX), (zx+2*BOX, zy+2*BOX), (255, 0, 0), 2)
    # Red box - Kalman 
    kx, ky = int(ukf.x[0]), int(ukf.x[1])
    cv2.rectangle(frame, (kx-BOX, ky-BOX), (kx+BOX, ky+BOX), (0, 0, 255), 2)
    cv2.imshow('Tracking', frame)
    
    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()

# Plotting
measurements = np.array(measurements)
estimates = np.array(estimates)
min_len = min(len(measurements), len(estimates))
measurements = measurements[:min_len]
estimates = estimates[:min_len]
plt.figure(figsize=(10, 6))   
# Blue box - Camera 
plt.plot(measurements[:,0], measurements[:,1], 'b.', label='Camera', alpha=0.3)
# Red box - Kalman 
plt.plot(estimates[:,0], estimates[:,1], 'r-', label='UKF', linewidth=2)
plt.gca().invert_yaxis() 
plt.title("UKF Test 2d")
plt.xlabel("X Pixel")
plt.ylabel("Y Pixel")
plt.legend()
plt.grid(True)
plt.show()
