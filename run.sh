pkill -f rollout_server
pkill -f buffer_server
python -m services.rollout_server --config config/dqn.yaml&
# python -m services.rollout_client
python -m services.buffer_server &
# python -m services.buffer_client
python main.py --config config/dqn.yaml
pkill -f rollout_server
pkill -f buffer_server