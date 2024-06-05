import cv2
import numpy as np
import sys

def track_surfboard(video_file, template_file):
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print("Error: Couldn't open the video file")
        return
    
    template = cv2.imread(template_file, 0)
    if template is None:
        print("Error: Couldn't load the template image")
        return
    
    template_h, template_w = template.shape[::-1]
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))
    
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    color = (0, 255, 255)
    
    surfboard_trajectory = []
    
    initial_detection = False
    gray_frame_prev = None
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not initial_detection:
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            
            if max_val > 0.8:  # Adjust threshold as necessary
                surfboard_x, surfboard_y = top_left
                initial_detection = True
                gray_frame_prev = gray_frame.copy()
                surfboard_trajectory.append((surfboard_x, surfboard_y))
                print(f"Initial detection at frame {frame_count}: ({surfboard_x}, {surfboard_y})")
            else:
                print(f"No initial detection at frame {frame_count}")
        else:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                gray_frame_prev, gray_frame, 
                np.float32([[surfboard_x, surfboard_y]]), None, **lk_params)
            
            if st[0][0] == 1:
                surfboard_x, surfboard_y = p1[0].ravel()
                surfboard_trajectory.append((surfboard_x, surfboard_y))
                print(f"Frame {frame_count}: Surfboard position: ({surfboard_x}, {surfboard_y})")
            
            for point in surfboard_trajectory:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, color, -1)
            
            gray_frame_prev = gray_frame.copy()
        
        out.write(frame)
        cv2.imshow('Surfboard Tracking', frame)
        
        frame_count += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    blank_image = np.zeros((frame_height, frame_width, 3), np.uint8)
    
    for point in surfboard_trajectory:
        cv2.circle(blank_image, (int(point[0]), int(point[1])), 3, color, -1)
    
    cv2.imwrite('final_trajectory.png', blank_image)

# Main function
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tracker.py <video_file> <template_file>")
    else:
        video_file = sys.argv[1]
        template_file = sys.argv[2]
        track_surfboard(video_file, template_file)
