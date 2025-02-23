pkill -f rollout_server
pkill -f buffer_server
python -m services.rollout_server --config config/ppo.yaml&
# python -m services.rollout_client
python -m services.buffer_server &
# python -m services.buffer_client
python main.py --config config/ppo.yaml
pkill -f rollout_server
pkill -f buffer_server