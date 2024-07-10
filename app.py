import argparse
from train.model_landmark import LandMarkModel
from train.model_process import load_model
from utils.game import AirCraft

# import time
import copy
# from utils import CvFpsCalc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proceed_score",type=int,default=10)
    parser.add_argument("--width", help='width', type=int, default=480*2)
    parser.add_argument("--height", help='height', type=int, default=800)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    parser.add_argument("--model_path",
                        help='model_path',
                        type=str,
                        default='model/save/model.pth')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model = LandMarkModel(42, output_size=1)
    model = load_model(model=model, file_path=args.model_path)
    game = AirCraft(screen_width=args.width,screen_height=args.height,model=model)
    game.game()

     
if __name__ == "__main__":
    main()