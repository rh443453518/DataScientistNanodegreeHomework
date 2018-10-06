from classification_model import *
from utils import *

def train():
    # get hyperparameters from command line arguments
    args = get_args_train()
    hyperparameters =  {'architecture': args.arch, 'epochs': args.epochs, 'print_every': args.print_every, 
                        'hidden_units': args.hidden_units, 'learning_rate': args.learning_rate, 
                        'dropout_prob': args.dropout_prob}

    # Call training function
    print('Start training image classification model.')
    model = train_classification_model(args.data_directory, args.gpu, hyperparameters)
    save_model(model, args.save_dir)
    
    
if __name__ == '__main__':
    train()