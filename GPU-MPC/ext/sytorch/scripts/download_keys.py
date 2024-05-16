# 
# Copyright:
# 
# Copyright (c) 2024 Microsoft Research
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
