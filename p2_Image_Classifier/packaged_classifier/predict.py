from classification_model import *
from utils import *
from PIL import Image

def predict():
    # get hyperparameters from command line arguments
    args = get_args_predict()
    
    # Load model, category_names and image
    print('Load model from checkpoint.')
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    device = 'cuda' if args.gpu else 'cpu'
    model = load_model(args.checkpoint, device)
    image = Image.open(args.path_to_image)
    
    # Process image to numpy array for prediction input
    np_image = process_image(image, model.architecture)
    
    # Make prediction of top k possible class with corresponding probabilities
    print('Start predicting top', args.top_k, 'possible image class.')
    top_k_probs, top_k_classes_name = predict_image_class(np_image, model, args.top_k, cat_to_name, device)
    print(top_k_classes_name)
    print(top_k_probs)


if __name__ == '__main__':
    predict()