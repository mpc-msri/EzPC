import ftplib
import logging

logging.basicConfig(level=logging.DEBUG)

ftp = ftplib.FTP()
ftp.connect("127.0.0.1", 9006)
ftp.login(user="client", passwd="client")

# # Download a file named "client.dat" from the server
# with open("client.dat", "wb") as f:
#     ftp.retrbinary("RETR client.dat", f.write)

# Now you can execute other FTP commands on the connected server
# For example, to list files in the current directory:
ftp.retrlines("LIST")

ftp.quit()
