import matplotlib.pyplot as plt

class validation_graphing():
    def validation_and_test_accuracy_graph(epochs, validation_accuracy, test_accuracy):
        # Data from the user's message
        #epochs = [1, 2, 3, 4, 5]
        #validation_accuracy = [0.312, 0.312, 0.875, 0.875, 0.875]
        #test_accuracy = [0.344, 0.281, 0.750, 0.750, 0.750]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='o')
        plt.plot(epochs, test_accuracy, label='Test Accuracy', marker='o')
        #plt.plot(epochs, train_loss, label='Test Accuracy', marker='o')
        plt.title('Epoch vs. Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def test_loss_graph(epochs, train_loss):
        # Data from the user's message
        #epochs = [1, 2, 3, 4, 5]
        #train_loss = [6.140, 4.689, 4.689, 4.624, 4.708]
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label='test_loss', marker='o')
        plt.title('Epoch vs. Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()