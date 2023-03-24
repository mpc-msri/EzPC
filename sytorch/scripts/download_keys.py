import requests
import time
import sys

# Define the URL of the file server and the file to download
url = sys.argv[1]
file_name = sys.argv[2]

# Define the size of each chunk to download
chunk_size = 1024 * 1024  # 1 MB

# Download the file repeatedly until a 200 status code is received
while True:
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Save the contents of the file to disk in chunks
            with open(file_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
            print("File downloaded successfully.")
            break
        else:
            # Wait for a short time before trying again
            print(
                f"Server returned status code {response.status_code}. Retrying in 10 seconds..."
            )
            time.sleep(10)
    except requests.exceptions.ConnectionError:
        # Wait for a short time before trying again
        print("Failed to connect to server. Retrying in 10 seconds...")
        time.sleep(10)
