from TuSimple_Testing import Skeletonize, ImgtoJson, Tusimple_metrics

def tusimple_eval(img_size, path_data, path_pred, save_path, pred_json_path,test_json_path):

    # Skeletonize the entire predicction
    print('Starting skeletonization')
    Skeletonize.skeletonize_dataset(img_size, path_data, path_pred, save_path)
    print('Done!')

    # Create Json
    print('Creating json')
    ImgtoJson.createJson(save_path,pred_json_path)
    print('Done!')

    # Evaluate
    print('Obtaining Tusimple reults')
    print(Tusimple_metrics.LaneEval.bench_one_submit(pred_json_path,test_json_path))


