echo "Waiting for debugger to attach..."
python -m debugpy --listen 10.66.20.1:1330 --wait-for-client train_dul_cls.py 