# System Status & Workflow Report

## 1. System Status
The FacePass system has been optimized and reorganized into two distinct sections:

1.  **Verification Kiosk (Home)**: A dedicated, public-facing screen for real-time face verification.
    *   **Features**: Full-screen video feed, live system status, real-time verified user counter.
    *   **Access**: Available at `http://localhost:8000/`

2.  **Admin Dashboard**: A restricted management area for system administrators.
    *   **Features**: User Registration, Employee Management, Attendance Logs.
    *   **Access**: Available at `http://localhost:8000/admin`

## 2. Multi-User Verification Workflow
The system is designed to handle multiple users simultaneously. Here is how it behaves when 2 or 3 employees appear on screen:

### Workflow
1.  **Detection**: The AI model scans the entire video frame 30+ times per second. It detects **all** visible faces, regardless of how many are present (limited only by hardware performance).
2.  **Processing**: The system loops through *each* detected face sequentially (but effectively instantly).
3.  **Verification**: 
    *   Face A is compared against the database -> Verified as "User A".
    *   Face B is compared against the database -> Verified as "User B".
    *   Face C is compared against the database -> Verified as "User C".
4.  **Attendance Logging**:
    *   If "User A" is verified, the system checks their last log time.
    *   If they haven't been logged in the last 30 seconds (Debounce), a **Check-In/Check-Out** event is recorded.
    *   This happens independently for "User B" and "User C".
5.  **Visual Feedback**:
    *   Each user gets a bounding box around their face.
    *   Their Name and ID are displayed above their head.
    *   Attendance notifications appear on screen for each verified user.

### Scenario: 3 Employees Walk In
*   **Result**: All 3 employees will be detected, recognized, and marked present almost simultaneously. There is no need for them to wait or stand one-by-one, provided their faces are clearly visible to the camera.

## 3. Best Practices for Deployment

### Camera Setup
*   **Position**: Mount the camera at eye level (approx. 1.6m - 1.7m) to capture frontal faces.
*   **Lighting**: Ensure consistent, frontal lighting. Avoid strong backlighting (windows behind users) or deep shadows.
*   **Resolution**: 720p or 1080p is ideal. Higher resolution improves accuracy but requires more CPU/GPU power.

### Registration (Critical)
*   **Pose Capture**: The new 5-pose registration flow is essential. Ensure users follow the instructions (Front, Left, Right, Up, Down) to build a robust face model.
*   **One Face Only**: Ensure only the person being registered is in the frame during the registration process.

### Hardware
*   **GPU Acceleration**: Essential for handling multi-face streams smoothly.
*   **Network**: If using IP cameras, ensure a stable wired connection to minimize latency.

## 4. Troubleshooting
*   **"Unknown" Face**: If a registered user shows as "Unknown", try re-registering them with better lighting or glasses removed.
*   **False Positives**: If the system confuses two people, the "Tolerance" threshold can be adjusted in the backend configuration (standard is 0.6, lower is stricter).
