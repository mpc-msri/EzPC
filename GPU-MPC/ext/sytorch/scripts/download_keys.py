import ftplib
import sys
from tqdm import tqdm
import time

url = sys.argv[1]
user = sys.argv[2]
passwd = sys.argv[3]
file_name = sys.argv[4]

while True:
    try:
        ftp = ftplib.FTP()
        ftp.connect(url, 9000)
        ftp.login(user=user, passwd=passwd)

        # Switch to binary mode
        ftp.sendcmd("TYPE i")

        # Get the size of the file on the server
        file_size = ftp.size(file_name)

        # Download the file and display a progress bar
        with open(file_name, "wb") as f:
            with tqdm(
                unit="B", unit_scale=True, unit_divisor=1024, total=file_size
            ) as progress:

                def callback(data):
                    f.write(data)
                    progress.update(len(data))

                ftp.retrbinary(f"RETR {file_name}", callback)

        ftp.quit()
        break  # exit the loop if the file is downloaded successfully

    except Exception as e:
        print(f"Error: Dealer not ready.")
        print("Retrying in 10 seconds...")
        time.sleep(10)
