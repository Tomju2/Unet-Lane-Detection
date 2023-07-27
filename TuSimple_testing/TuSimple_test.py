import Skeletonize, ImgtoJson, Tusimple_metrics
from loguru import logger
import argparse
from pathlib import Path

"""Tusimple official evaluation

Args:
    img_size (int): size of the images
    path_data (string): label images path
    path_pred (string): inference images from the model
    save_path (string): where the processed images will be saved
    pred_json_path (string): Path where the json will be saved
    test_json_path (string): path of the official tusimple jsons
"""

parser = argparse.ArgumentParser()
parser.add_argument("path_data")
parser.add_argument("path_pred")
parser.add_argument("save_path")
parser.add_argument("pred_json_path")
parser.add_argument("test_json_path")
args = parser.parse_args()

for element in vars(args):
    arg_test_path = Path(getattr(args, element))
    if not arg_test_path.exists():
        logger.error("The " + element + " dosen't exist")
        raise SystemExit(1)

path_data = Path(args.path_data)
path_pred = Path(args.path_pred)
save_path = Path(args.save_path)
pred_json_path = Path(args.pred_json_path)
test_json_path = Path(args.test_json_path)

# Skeletonize the entire predicction
logger.info('Starting skeletonization')
Skeletonize.skeletonize_dataset(256, path_data, path_pred, save_path)
logger.info('Done!')

# Create Json
logger.info('Creating json')
ImgtoJson.createJson(save_path,pred_json_path)
logger.info('Done!')

# Evaluate
logger.info('Obtaining Tusimple reults')
print(Tusimple_metrics.LaneEval.bench_one_submit(pred_json_path,test_json_path))


