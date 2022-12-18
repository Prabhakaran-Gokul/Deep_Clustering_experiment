from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model
from DEKM import model_conv, loss_train_base, dense_model_conv
from utils import get_xy

def get_predictions(model_path='dense_weight_final_l2_MNIST.h5'):
    # load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # load model from final weights
    model = dense_model_conv(False)
    print(model.summary())
    # model.load_weights(model_path)
    plot_model(model=model, show_shapes=True, to_file="dense_model.png")
    # model.compile(optimizer='adam', loss=loss_train_base)

    # model.evaluate(x_test, y_test, verbose=1) #83.5986
    # predictions = model.predict(x_test)
    # print(predictions[0], predictions[0].shape)

if __name__ == "__main__":
    get_predictions()
    # x, y = get_xy("MNIST")
    # print(x.shape, y.shape)