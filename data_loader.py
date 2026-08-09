import os
import cv2
import glob

def frame_generator(path):
    """
    Yields frames one by one to prevent RAM overload.
    Supports either a directory of images (PNG/JPG) or a standard video file (.mp4, .avi).
    """
    if os.path.isfile(path):
        # Handle video file
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error opening video stream or file {path}")
            return
            
        print(f"Streaming video from {path}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale as required by processing pipeline
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield gray_frame
            
        cap.release()
        
    elif os.path.isdir(path):
        # Handle directory of images
        # Check standard images subfolder first
        search_path_png = os.path.join(path, 'images', '*.png')
        search_path_jpg = os.path.join(path, 'images', '*.jpg')
        image_files = sorted(glob.glob(search_path_png) + glob.glob(search_path_jpg))
        
        if not image_files:
            # Check root directory
            search_path_png = os.path.join(path, '*.png')
            search_path_jpg = os.path.join(path, '*.jpg')
            image_files = sorted(glob.glob(search_path_png) + glob.glob(search_path_jpg))
            
        print(f"Found {len(image_files)} image frames in {path}")
        
        for img_path in image_files:
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                yield frame
    else:
        print(f"Path {path} is neither a valid file nor a directory.")
