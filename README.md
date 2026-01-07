#  Drone State Estimation: UKF vs. Standard Kalman Filter

##  Overview
This project implements and compares two advanced state estimation algorithms for real-time drone tracking:
1.  **Standard Kalman Filter (KF):** Best for linear motion.
2.  **Unscented Kalman Filter (UKF):** Handles non-linear trajectories and complex maneuvers.

The system fuses noisy **Computer Vision** measurements with **Control Theory** models to estimate the drone's position and velocity with high precision.

##  Key Features
* **Algorithm Comparison:** A side-by-side performance analysis of Linear KF vs. UKF in tracking maneuvering targets.
* **Robust Tracking:** Successfully filters out sensor noise ($R$) and accounts for process uncertainty ($Q$).
* **Computer Vision Integration:** Bridging the gap between raw pixel data (detection) and state estimation.
* **Real-Time Visualization:** Visual plotting of the estimated trajectory against raw noisy measurements.

##  Tech Stack & Concepts
* **Languages:** Python 3.x (`NumPy`, `OpenCV`, `Matplotlib`)
* 
##  How It Works
1.  **Detection:** The system extracts the drone's coordinates from video frames (simulating a visual sensor).
2.  **Prediction:** The filter predicts the next state based on a physical motion model (Constant Velocity).
3.  **Correction (Update):**
    * The **Standard KF** applies linear matrix gains.
    * The **UKF** utilizes **Sigma Points** (Unscented Transform) to propagate probability densities through non-linear functions, providing superior stability in turns.

## 📊 Results
The graph below demonstrates the UKF's ability to smooth the noisy camera data (Blue) into a coherent flight path (Red), significantly reducing jitter.

![UKF Results](Figure_1.png)


