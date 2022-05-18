import os
from DNN import DNN
from datetime import datetime
from keras.losses import MeanSquaredError

def get_params():
    input_size = 512
    num_filters = 64
    kernel_size = 3
    activation = 'relu'
    kernel_regularizer = None
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    model_name = f"DNN_CV_{input_size}x{input_size}_{timestamp}"

    return input_size, num_filters,kernel_size,activation,kernel_regularizer,model_name

def getTrainingParams():
    initial_lr = 0.0001
    loss = MeanSquaredError()
    metrics = ['accuracy']
    test_split = 0.2
    random_state = 13412
    loss = 'MSE'

    return initial_lr,loss,metrics, test_split, random_state,loss
def main():
    cnn_dir = os.getcwd()
    dataset_folder = 'dataset_1_test'
    dataset_dir = os.path.join(cnn_dir, dataset_folder)
    input_size, num_filters, kernel_size, activation, kernel_regularizer, model_name = get_params()
    initial_lr, loss, metrics, test_split, random_state, loss = getTrainingParams()
    DNN_40x40 = DNN(input_size, num_filters,kernel_size,activation,kernel_regularizer,model_name, cnn_dir)

    model = DNN_40x40.getModel()
    DNN_40x40.compile(initial_lr,loss,metrics,summarize=True, tofile=True)
    DNN_40x40.plot_results(trained=False)
    DNN_40x40.save_log(test_split, random_state, dataset_dir, trained=False)




if __name__ == '__main__':
    main()