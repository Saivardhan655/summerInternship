import os
import pandas as pd
import tensorflow as tf

# Example function to extract features from a video (stub, replace with your implementation)
def extract_features(video_file):
    # Replace this stub with your actual feature extraction logic
    # Here, I return dummy data for demonstration purposes
    return [[0.5, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]  # Replace with actual feature extraction logic

def save_features(video_files):
    print("Save feature called")
    all_features = []
    
    for video_file in video_files:
        print(f"Processing video: {video_file}")
        features = extract_features(video_file)
        print(f"Features collected for {video_file}: {features}")
        all_features.extend(features)

    print(f"Total number of features collected: {len(all_features)}")

    if len(all_features) > 0:
        # Convert to DataFrame
        df = pd.DataFrame(all_features, columns=['EAR', 'MAR', 'Emotion1', 'Emotion2', 'Emotion3', 'Emotion4', 'Emotion5', 'Emotion6', 'Emotion7'])
        print("DataFrame shape:", df.shape)
        print("DataFrame head:", df.head())  # Debug line
        
        output_file = "C:/Users/DELL/summerInternship/data/labeled_features.csv"
        try:
            # Check if output directory exists
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                print(f"Output directory does not exist. Creating directory: {output_dir}")
                os.makedirs(output_dir)

            print(f"Attempting to save features to {output_file}...")
            df.to_csv(output_file, index=False)
            print(f"Features saved to CSV at {output_file}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    else:
        print("No features collected; nothing to save.")

if __name__ == "__main__":
    # Example list of video files to process (replace with your actual video file paths)
    video_files = [
        "C:/Users/DELL/summerInternship/data/videos/video1.mp4",
        "C:/Users/DELL/summerInternship/data/videos/video2.mp4",
        # Add more video paths as needed
    ]

    # Call the function to save features
    save_features(video_files)
