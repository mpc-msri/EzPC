# This file contains settings that must be changed for each deployment

# The name that will be shown to users
CONTEST_NAME = "Imagenet Inference Contest"

# Number of testcases each submission will be evaluated on
NUM_TESTCASES = 5

# Number of classes. For example, for imagenet this will be 1000. For cat/dog inference it will be 2
NUM_CLASSES = 1000

# The public IP address of the server where the website will be located
WEBSITE_IP_ADDRESS = "127.0.0.1"

# The port in which website will be run. You need to start django using `python manage.py runserver WEBSITE_PORT``
WEBSITE_PORT = 8000


# Set to false in production
DEBUG_MODE = True

# Change in production to a new random key
SECRET_CONFIG_KEY = "django-insecure-cnmv0)-2-=v7tro^nt@t2%=eqgvvzmp%ifw2$(*9g!lj5(!os4"
