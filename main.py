import yaml
import argparse
from pprint import pprint
from os.path import join

from utils import load_graph, load_ids
from controllers.controllers_netact import NetActController


def main(args):
    with open(args.param_file) as read_file:
        configs = yaml.load(read_file)
    # pprint(configs)
    user_id, reverse_user_id, item_id, reverse_item_id = \
        load_ids(configs['data']['dataset_dir'], configs['data']['ids_file_name'])
    adj_mat = load_graph(join( configs['data']['dataset_dir'], configs['data']['graph_file_name']), len(user_id))

    if configs['model_name'] == 'netact':
        mc = NetActController(adj_mat, **configs)
        mc.run_train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', default='params.yaml', type=str, help='parameter file')
    args = parser.parse_args()
    main(args)
