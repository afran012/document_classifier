import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py [train|predict]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        os.system("python scripts/train_model.py")
    elif command == "predict":
        os.system("python scripts/predict_first_pages.py")
    else:
        print("Comando no reconocido. Uso: python main.py [train|predict]")
        sys.exit(1)

if __name__ == "__main__":
    main()