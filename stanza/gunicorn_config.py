# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 2  # Adjust based on your CPU cores
threads = 4
timeout = 120
worker_class = "gthread"  # Use threads for better concurrency with Stanza
