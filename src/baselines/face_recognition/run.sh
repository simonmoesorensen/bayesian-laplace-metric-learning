echo "Waiting for debugger to attach..."
python -m debugpy --listen 10.66.20.1:1331 --wait-for-client main.py