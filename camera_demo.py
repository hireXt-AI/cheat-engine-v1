import argparse
import logging
import sys
from pathlib import Path

from face_detection import match_reference_image_in_camera

# Set up logging for clean terminal output
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

def main():
    parser = argparse.ArgumentParser(
        description="Demo for matching a reference image against live camera frames."
    )
    
    # Use type=Path to handle file paths automatically across Windows/Mac/Linux
    parser.add_argument(
        "reference", 
        type=Path, 
        help="Path to the reference image file."
    )
    parser.add_argument(
        "--camera-index", 
        type=int, 
        default=0, 
        help="Camera index to use (default: 0)."
    )
    parser.add_argument(
        "--match-threshold", 
        type=float, 
        default=0.25, 
        help="Face similarity threshold (default: 0.25)."
    )
    parser.add_argument(
        "--min-good-matches", 
        type=int, 
        default=10, 
        help="Minimum number of good feature matches (default: 10)."
    )
    
    args = parser.parse_args()

    # 1. Pre-validate that the file actually exists
    if not args.reference.is_file():
        logging.error(f"Reference image not found at: {args.reference.absolute()}")
        sys.exit(1)

    logging.info(f"Starting camera {args.camera_index} using reference: {args.reference.name}")
    logging.info("Press 'Ctrl+C' in terminal or 'q' in the video window to exit.")

    # 2. Wrap execution in a try-except block for graceful exits
    try:
        match_reference_image_in_camera(
            reference_path=str(args.reference), # Convert Path back to string for OpenCV
            camera_index=args.camera_index,
            match_threshold=args.match_threshold,
            min_good_matches=args.min_good_matches,
        )
    except KeyboardInterrupt:
        # Catch Ctrl+C to exit cleanly
        print() 
        logging.info("Camera feed stopped by user (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        # Catch webcam lockouts or unexpected OpenCV crashes
        logging.error(f"An error occurred during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()