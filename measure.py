import os
from scripts.measure_tree import run
from utils.log_utils import parser_measure_args

if __name__ == '__main__':
    args = parser_measure_args()

    data_path = args.data_path
    info= {'data_dir':[], 'output_dir':[], 'split_json':[]}
    for seed in os.listdir(data_path):
        info['data_dir'].append(os.path.join(data_path, seed, 'swc'))
        info['output_dir'].append(os.path.join(data_path, seed))
        number_seed = seed.split('_')[-1]
        info['split_json'].append(os.path.join(data_path, seed, f'{number_seed}.json'))

    run(info, measure_name=args.measure_type, is_morph=args.is_morph)