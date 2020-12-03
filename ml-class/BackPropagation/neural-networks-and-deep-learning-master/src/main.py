from src.mnist_loader import load_data_wrapper
from src.network import Network

def main():
    training_data, validation_data, test_data = load_data_wrapper()
    network = Network([784, 15, 10])
    network.SGD(training_data=training_data, epochs=30, mini_batch_size=64, eta=0.5, test_data=test_data)

main()


