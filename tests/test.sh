work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
CUDA_VISIBLE_DEVICES='0' python test.py -exp test_human --gpu_id 0 --beta 0  -c /path/to/checkpoint -d /path/to/dataset
CUDA_VISIBLE_DEVICES='0' python test.py -exp test_machine --gpu_id 0 --beta 1  -c /path/to/checkpoint -d /path/to/dataset

