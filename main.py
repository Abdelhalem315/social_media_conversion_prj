import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Social Media Conversion Prediction Pipeline")
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'predict', 'app'], 
        default='predict',
        help="Choose pipeline execution mode: 'train' to retrain model, 'predict' to run inference test, or 'app' to launch Streamlit UI."
    )

    args = parser.parse_args()

    if args.mode == 'train':
        print("Starting Training Pipeline...")
        os.system("python train.py")
        
    elif args.mode == 'predict':
        print("Running Inference Pipeline Test...")
        os.system("python predict.py")
        
    elif args.mode == 'app':
        print("Launching Streamlit Web App...")
        os.system("streamlit run app.py")

if __name__ == "__main__":
    main()










