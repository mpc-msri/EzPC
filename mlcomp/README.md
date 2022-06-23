# mlcomp

A system for automating evaluation of MPC ONNX models in a contests format. This is a Django Application.

This is called the "website". Contest "organizer"s and "participant"s need their own VMs to use this system.

## Learn Django

To understand how this works, complete the Django Tutorial at https://docs.djangoproject.com/en/4.0/contents/.

## Participant and Organizer automation scripts

The participant automation script is in `contests/static/contests/scripts/participant_script.py`

The organizer automation script is in `contests/static/contests/scripts/organizer_script.py`.

The label comparison step compiled binary is in `contests/static/contests/objects/compare_labels_nm` and the source is in `contests/static/contests/objects/compare_labels_nm`.

## Setup

1. Setup the modified EzPC code. Alternatively, you can use this modified Docker image: https://hub.docker.com/r/agrawald/ezpc-modified.


```
git clone https://github.com/msri-mpc-d/EzPC
git checkout mlcomp
git pull origin mlcomp
cd EzPC
./setup_env_and_build.sh quick
```

2. Clone this repository and `cd` into it.

```
cd ..
git clone https://github.com/mpc-msri/mlcomp
cd mlcomp
```

3. Activate the EzPC venv using `source path/to/EzPC/repo/mpc_venv/bin/activate`

```
source ../EzPC/mpc_venv/bin/activate
```

4. Install dependencies.

```
pip install -r requirements.txt
```

5. In settings.py change `WEBSITE_URL` to the IP address where this will be hosted ( keep it same for local development )
6. You may need to change `ALLOWED_HOSTS` and `SECRET_KEY` in settings.py in production. See Django documentation for details.
7. Run migrations

```
python manage.py migrate && python manage.py migrate contests
```

8. Create the superuser using

```
python manage.py createsuperuser
```

9. Run the development server

```
python manage.py runserver
```

10. You can access the website at http://127.0.0.1.
11. You can access the Django Admin Portal at http://127.0.0.1/admin.

## Create a contest

The guide to create a sample contest is [here](guide.md)