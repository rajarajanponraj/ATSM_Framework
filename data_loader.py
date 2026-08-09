import os
import cv2
import glob

def frame_generator(sequence_dir):
    """
    Yields frames one by one from a sequence directory to prevent RAM overload.
    DAVIS dataset usually contains 'images' folder with PNG files.
    """
    search_path = os.path.join(sequence_dir, 'images', '*.png')
    image_files = sorted(glob.glob(search_path))
    
    if not image_files:
        # Fallback if there is no 'images' subfolder
        search_path = os.path.join(sequence_dir, '*.png')
        image_files = sorted(glob.glob(search_path))
        
    print(f"Found {len(image_files)} frames in {sequence_dir}")
    
    for img_path in image_files:
        # Load in grayscale as required by most of our processing
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            yield frame
