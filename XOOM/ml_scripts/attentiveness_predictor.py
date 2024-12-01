import base64
import sys
import os
import json

def main():
    # Read the file path passed from Node.js
    file_path = sys.argv[1]

    try:
        # Open the file and read the image data
        with open(file_path, 'rb') as file:
            img_data = file.read()

        # Process the image data (e.g., perform prediction)
        # This should be replaced with actual model inference code
        result = {
            "attentiveness": 0.85  # Replace with actual result from model
        }
        
        # Output the result as a valid JSON string
        print(json.dumps(result))  # Ensures the result is JSON
        
        # Remove the temporary file after processing
        os.remove(file_path)

    except Exception as e:
        # In case of error, print an error message as JSON
        error_result = {
            "error": str(e)
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    main()
