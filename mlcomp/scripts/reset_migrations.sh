#!/bin/bash
echo 'Resetting migrations and deleting db'
rm -rf contests/migrations
rm -v db.sqlite3
python manage.py makemigrations
python manage.py makemigrations contests
python manage.py migrate
python manage.py migrate contests
clear
echo 'Done'
