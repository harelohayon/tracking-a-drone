import cv2
import numpy as np
import matplotlib.pyplot as plt
# פונקציה שמזהה את הנקודה הכי בהירה בתמונה (כלומר את מקור האור)
def detect_object(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # המר לתמונה בגווני אפור
    a,b,c,maxLoc = cv2.minMaxLoc(gray)                    # מצא את הפיקסל הבהיר ביותר
    return maxLoc                                   # מחזיר (x, y) של הנקודה הכי בהירה
# פתיחת הסרטון
cap = cv2.VideoCapture(r"C:\Users\HARIEL\OneDrive\Desktop\light.mp4")
if  cap.isOpened():
    print('yes')
else:
    print("no")
# הגדרת פילטר קלמן
kf = cv2.KalmanFilter(4, 2)  # שני משתני מדידה וארבע משתני מצב
w = kf.processNoiseCov  # רעש התהליך Q 
h = kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # מטריצת המדידה H 
dt = 1
f = kf.transitionMatrix = np.array([[1, 0, dt, 0],[0, 1, 0, dt],[0, 0, 1,  0], [0, 0, 0,  1]], np.float32)

p = kf.errorCovPost = np.eye(4, dtype=np.float32)  # מטריצת קו-וריאנציה של שגיאת ההתחלהה
kf.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32)  # מצב התחלתי של פילטר קלמן – מיקום ומהירות אפס

# איסוף מיקומים מהסרטון
positions = []
findobject, frame = cap.read()  # קריאת הפריים הראשון
while findobject == True:   
    x, y = detect_object(frame) 
    positions.append((x, y))
    findobject, frame = cap.read()  # קריאה לפריים הבא
cap.release()  # סיום קריאת הסרטון

# המשטח שבו יצויר הגרף
canvas_height, canvas_width = 400, 600
frame_sequence = []

estimates = []

for i in range(len(positions)):
    z = positions[i]  # המדידה הנוכחית מהסרטון
    measured = np.array([[np.float32(z[0])],
                         [np.float32(z[1])]])

    kf.correct(measured)          # שלב התיקון – כולל חישוב Kalman Gain ועדכון המצב והקו-וריאנציה
    prediction = kf.predict()     # שלב התחזית – חיזוי המיקום הבא לפי המהירות הנוכחית

    pred_x, pred_y = int(prediction[0,0]), int(prediction[1,0])  # ערכים של התחזית לציור (שלמים)
    est_x, est_y = prediction[0,0], prediction[1,0]              # אותם ערכים לשמירה (float)
    estimates.append([est_x, est_y])

    # ציור הפריים
positions = np.array(list(positions), dtype=np.float32).reshape(-1, 2)
estimates = np.array(estimates, dtype=np.float32).reshape(-1, 2)
plt.figure(figsize=(10, 6))

# מדידות מתוך הסרטון – כחול
plt.plot(positions[:, 0], positions[:, 1], 'bo-', label='Measured (from video)')

# תחזיות של פילטר קלמן – אדום
plt.plot(estimates[:, 0], estimates[:, 1], 'ro-', label='Kalman Estimate')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Kalman Filter Tracking")
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()  # אם הצירים הפוכים כמו במצלמה
plt.show()