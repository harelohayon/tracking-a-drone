# 🚁 Drone State Estimation: UKF vs. Standard Kalman Filter

## 📌 Overview
This project implements and compares two advanced state estimation algorithms for real-time drone tracking:
1.  **Standard Kalman Filter (KF):** Optimized for linear motion.
2.  **Unscented Kalman Filter (UKF):** Designed to handle non-linear trajectories and complex maneuvers.

The system fuses noisy **Computer Vision** measurements with **Control Theory** models to estimate the drone's position and velocity with high precision.

## ⚙️ Key Features
* **Algorithm Comparison:** A side-by-side performance analysis of Linear KF vs. UKF in tracking maneuvering targets.
* **Robust Tracking:** Successfully filters out sensor noise ($R$) and accounts for process uncertainty ($Q$).
* **Computer Vision Integration:** Bridges the gap between raw pixel data and physical state estimation.
* **Real-Time Visualization:** Visual plotting of the estimated trajectory against raw noisy measurements.

## 🛠️ Tech Stack & Concepts
* **Languages:** Python 3.x (`NumPy`, `OpenCV`, `Matplotlib`)
* **Core Domains:**
    * **Control Theory:** State Estimation, Sensor Fusion, Covariance Analysis.
    * **Computer Vision:** Object Detection pipeline.
    * **Embedded Systems:** Efficient matrix operations designed for real-time constraints.

## 🚀 How It Works
1.  **Detection:** The system extracts coordinates from video frames.
    * *Note:* Currently, the system tracks the drone's **LED marker** to simulate a visual sensor. This isolates the tracking algorithm performance from detection errors.
2.  **Prediction:** The filter predicts the next state based on a physical motion model (Constant Velocity).
3.  **Correction (Update):**
    * The **Standard KF** applies linear matrix gains.
    * The **UKF** utilizes **Sigma Points** (Unscented Transform) to propagate probability densities through non-linear functions, providing superior stability during turns.

## 📊 Results
The graph below demonstrates the UKF's ability to smooth the noisy camera data (**Blue**) into a coherent flight path (**Red**), significantly reducing jitter.

![Graph Results](Figure_1.png)

## 💻 Usage
Since the project compares two methods, the code is split into independent modules:

**To run the Unscented Kalman Filter (Recommended):**
```bash
python ukf_drone_tracking.py
