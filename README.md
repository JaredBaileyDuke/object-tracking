# Tracking within Images

## ArUco Markers

#### Files for Use in Assets Folder
- aruco_markers_0.pdf
- aruco_markers_1.pdf

#### Run Script
```bash
python src/aruco_markers.py
```

To change camera, change the python script number:
cap = cv2.VideoCapture(0)

Press 'q' to close the pop-up window


## Object Tracking

#### Files for Use in Assets Folder
- plane_short.mp4


#### Run Script
```bash
python src/object_tracking.py
```

Look for the pop-up in order to select your object to track


## Optical Flow

#### Files for Use in Assets Folder
- alligator_short.mp4

#### Run Script
```bash
python src/optical_flow_lk.py
python src/optical_flow_f.py
```