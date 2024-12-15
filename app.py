from flask import Flask, jsonify, render_template, send_from_directory
import logging
import threading
import cv2
import numpy as np
import smtplib
import os
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Suppress HTTP request logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'extracted_vehicle_images'
vehicle_data = {}

summary_data = {
    "vehicles_down": 0,
    "vehicles_up": 0,
    "traffic_density": 0.0,
    "total_vehicle_count": 0,
}

# Endpoint to get live data
@app.route('/data')
def data():
    return jsonify(summary_data)

# Endpoint to get vehicle details
@app.route('/vehicle_data')
def vehicle_data_api():
    return jsonify(list(vehicle_data.values()))

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

def update_summary_data(down, up, density, total_count):
    summary_data["vehicles_down"] = down
    summary_data["vehicles_up"] = up
    summary_data["traffic_density"] = density
    summary_data["total_vehicle_count"] = total_count

# Start Flask server
def run_flask():
    app.run(debug=False, use_reloader=False)

# Start Flask in a separate thread
threading.Thread(target=run_flask).start()
print("\033[1;32mFlask server is running. Access it at \033[1;34mhttp://127.0.0.1:5000\033[0m")

# Capture video from file or camera
cap = cv2.VideoCapture("video.mp4")

# Minimum rectangle size to filter small contours
min_width_rect = 80
min_height_rect = 80

# Line position for counting vehicles
count_line_position = 550
offset = 6  # Allowable error for line crossing

# Initialize the background subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# Function to calculate the center of a bounding box
def center_handle(x, y, w, h):
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy

# Counter for total vehicles and lists for vehicle IDs
total_counter = 0
counter_up = 0
counter_down = 0
vehicle_ids_up = []  # Store IDs for vehicles going up
vehicle_ids_down = []  # Store IDs for vehicles going down

# Check if video opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Use the actual frame rate of the video
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate from the video file
print(f"Frame Rate: {frame_rate} FPS")

# Adjust this based on a known distance in the scene
meters_per_pixel = 0.05  # Example value; adjust as necessary

# Distance threshold for associating vehicle positions in consecutive frames
distance_threshold = 50
previous_centers = []  # To store previous frame centers for comparison

# Email details
email = "utsavtyagi3456@gmail.com"  # Your email
receiver_email = "hanutyagi156@gmail.com"  # Receiver's email
subject = "Overspeeding Alert!"  # Email subject

# Create directory for storing vehicle images
output_dir = app.config['UPLOAD_FOLDER']
os.makedirs(output_dir, exist_ok=True)

# Set to track saved vehicle IDs
saved_vehicle_ids = set()

# Function to send email alerts with vehicle ID and speed
def send_email_alert(vehicle_id, speed):
    try:
        # Create an Excel file with vehicle details
        create_excel_report()

        message = f"A vehicle with ID {vehicle_id} is overspeeding at {speed:.2f} m/s. Please check the details of the Vehicle here"
        text = f"Subject: {subject}\n\n{message}"
        
        # Specify the attachment file
        attachment = "vehicle_details_updated.xlsx"
        
        # Setting up the email server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls() 

        # Login with your email and app password
        server.login(email, "mjcn swvz dcnr kelv")  # Replace with your app password

        msg = MIMEMultipart()
        msg['From'] = email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        # Attach the Excel file
        with open(attachment, "rb") as file:
            part = MIMEApplication(file.read(), Name=os.path.basename(attachment))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment)}"'
            msg.attach(part)

        # Send the email
        server.send_message(msg)
        blue = "\033[34m"
        reset = "\033[0m"
        print(f"{blue}Email alert sent to {receiver_email} for vehicle ID {vehicle_id} overspeeding at {speed:.2f} m/s.{reset}")
        server.quit()
    except Exception as e:
        print("Error sending email:", e)

def create_excel_report():
    # Create a DataFrame with vehicle details
    data = {
        "Vehicle ID": range(1, total_counter + 1),
        "Speed (m/s)": [np.random.uniform(20, 50) for _ in range(total_counter)],  # Placeholder for speed values
        "Direction": ["Up" if i in vehicle_ids_up else "Down" for i in range(1, total_counter + 1)]
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel("vehicle_details_updated.xlsx", index=False)

while cap.isOpened():
    ret, frame1 = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Applying background subtractor on each frame
    img_sub = algo.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)), iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate_data = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(dilate_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw counting line
    cv2.line(frame1, (170, 450), (1050, 450), (236, 18, 252), 3)
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    current_centers = []  # Store the centers detected in the current frame

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= min_width_rect and h >= min_height_rect:
            center = center_handle(x, y, w, h)
            current_centers.append(center)

            # Draw rectangle and center point on the vehicle
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)

            # Initialize vehicle_id to None for this contour
            vehicle_id = None

            # Compare current center with previous centers
            matched = False
            for prev_center in previous_centers:
                dist = np.linalg.norm(np.array(center) - np.array(prev_center['position']))

                # Check if it's the same vehicle crossing the line
                if dist < distance_threshold:
                    matched = True

                    # Vehicle crosses the line going up
                    if prev_center['position'][1] > count_line_position >= center[1]:
                        counter_up += 1
                        total_counter += 1  # Increment total counter
                        vehicle_id = total_counter  # Assign new vehicle ID
                        vehicle_ids_up.append(vehicle_id)  # Store ID for upward vehicle
                        cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 255, 255), 3)

                    # Vehicle crosses the line going down
                    elif prev_center['position'][1] <= count_line_position and center[1] > count_line_position:
                        counter_down += 1
                        total_counter += 1  # Increment total counter
                        vehicle_id = total_counter  # Assign new vehicle ID
                        vehicle_ids_down.append(vehicle_id)  # Store ID for downward vehicle
                        cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)

                    # Calculate speed
                    # Use the previous center's position to estimate speed
                    prev_position = prev_center['position']
                    distance_moved = np.linalg.norm(np.array(center) - np.array(prev_position)) * meters_per_pixel
                    time_elapsed = 1 / frame_rate  # Time per frame in seconds
                    speed = distance_moved / time_elapsed  # Speed in meters per second

                    # Display vehicle ID and speed in the terminal if ID is assigned
                    if vehicle_id is not None:
                        print(f"Vehicle ID: {vehicle_id}, Current Speed: {speed:.2f} m/s")

                        # Check for overspeeding
                        if speed > 30:
                            print(f"\033[1;31m!!! ALERT: Vehicle ID : {vehicle_id} is OVERSPEEDING at {speed:.2f} m/s !!!\033[0m")

                            # Send email alert in a separate thread
                            threading.Thread(target=send_email_alert, args=(vehicle_id, speed)).start()

                        # Extract the vehicle's image and save it if not already saved
                        if vehicle_id not in saved_vehicle_ids:
                            vehicle_image = frame1[y:y + h, x:x + w]  # Extract vehicle image
                            cv2.imwrite(os.path.join(output_dir, f"vehicle_id_{vehicle_id}.jpg"), vehicle_image)
                            saved_vehicle_ids.add(vehicle_id)  # Mark this ID as saved
                            vehicle_data[vehicle_id] = {
                                "id": vehicle_id,
                                "speed": speed,
                                "image_path": f"/images/vehicle_id_{vehicle_id}.jpg"
                            }
                    
                    # Update position for the matched vehicle in the previous_centers list
                    prev_center['position'] = center
                    break

            # If no match is found, treat as new detection
            if not matched:
                previous_centers.append({'id': total_counter + 1, 'position': center})

            # Display vehicle ID on the frame if it was assigned
            if vehicle_id is not None:
                cv2.putText(frame1, f"ID: {vehicle_id}", (center[0] - 10, center[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)

    # Calculate traffic density
    num_vehicles = counter_up + counter_down
    frame_area = frame1.shape[0] * frame1.shape[1]  # Total area of the frame in pixels
    density = num_vehicles / (frame_area / (10000))  # Vehicles per square meter (assuming each pixel = 0.01 m²)
    update_summary_data(counter_down, counter_up, density, total_counter)
    
    # Update previous centers for the next frame
    previous_centers = [{'id': prev['id'], 'position': center} for prev, center in zip(previous_centers, current_centers) if center in current_centers]

    # Display vehicle counts on the frame
    cv2.putText(frame1, f"Vehicles Going Down: {counter_down}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame1, f"Vehicles Going Up: {counter_up}", (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    # Display traffic density
    cv2.putText(frame1, f"Traffic Density: {density:.2f}", (450, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
    cv2.putText(frame1, f"vehicles/m^2", (1000, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2 )

    # Show the current frame
    cv2.imshow("Vehicle Detection in Traffic Management", frame1)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# ANSI escape code for green color
bold_green = "\033[1;92m"
reset = "\033[0m"  # Reset to default color

# Final print statements for vehicle count
print(f"\n{bold_green}-----------------------  SUMMARY OF THE VEHICLE DETECTION  --------------------------")
print(f"\n{bold_green}Total Vehicles Moving Down: {reset}{counter_down}")
print(f"\n{bold_green}Total Vehicles Moving Up: {reset}{counter_up}")
print(f"\n{bold_green}Vehicle IDs Moving Up: {reset}{vehicle_ids_up}")
print(f"\n{bold_green}Vehicle IDs Moving Down: {reset}{vehicle_ids_down}")
print(f"\n{bold_green}Total Vehicle IDs: {reset}{total_counter}")
print(f"\n{bold_green}Total Vehicle Count: {reset}{total_counter}")
print(f"\n{bold_green}Traffic Density: {reset}{density:.2f}")
