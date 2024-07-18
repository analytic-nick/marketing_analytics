# Write the Procfile
# - app refer to the file name app.py
# - server refer to the variable name for the Flask Server 
web: gunicorn app:server