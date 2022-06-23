from django.contrib import admin
from django.contrib.auth.models import User
from contests.models import Submission

# Register your models here.

admin.site.register(Submission)
