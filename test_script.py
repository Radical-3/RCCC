from method import (training_camouflage, dataset_cleaning,
                    test_camouflage, track, track2, track3,
                    track4, track5, test_camera_position, test_camera_position,
                    test_camera_position2, track_result, evaluate_tracking_results)

if __name__ == "__main__":
    # track4()
    # track5()
    # test_camera_position()
    # test_camera_position2()
    # track_result()
    evaluate_tracking_results("./output/camo_metrics/12-24/predict_bbox", "./output/camo_metrics/ori_bbox")

