kill $(ps aux | grep '-m debugpy' | awk '{print $2}')
