import numpy as np 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# prétraitement : conversion des images en Tenseurs normalisés entre 0 et 1
transform = transforms.ToTensor()

class Model:
    """réseau de neurones multicouche fait entièrement en NumPy.
        architecture : 784 (Entrée) -> 128 -> 64 -> 32 -> 16 -> 10 (Sortie Softmax)
    """
    def __init__(self):  
        #  initialisation de poids (He Initialization)
        # On utilise He Init pour eviter la disparition/explosion du gradient avec ReLU

        self.w1 = np.random.randn(784, 128) * np.sqrt(2/784)
        self.b1 = np.zeros(128)

        self.w2 = np.random.randn(128, 64) * np.sqrt(2/128)
        self.b2 = np.zeros(64)

        self.w3 = np.random.randn(64, 32) * np.sqrt(2/64)
        self.b3 = np.zeros(32)

        self.w4 = np.random.randn(32, 16) * np.sqrt(2/32)
        self.b4 = np.zeros(16)

        self.w5 = np.random.randn(16, 10) * np.sqrt(2/16)
        self.b5 = np.zeros(10)

        # MOMENTUM (Optimisation) 
        self.v_w1, self.v_b1 = np.zeros_like(self.w1), np.zeros_like(self.b1)
        self.v_w2, self.v_b2 = np.zeros_like(self.w2), np.zeros_like(self.b2)
        self.v_w3, self.v_b3 = np.zeros_like(self.w3), np.zeros_like(self.b3)
        self.v_w4, self.v_b4 = np.zeros_like(self.w4), np.zeros_like(self.b4)
        self.v_w5, self.v_b5 = np.zeros_like(self.w5), np.zeros_like(self.b5)

        self.lr = 0.01

    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def loss(self, y_pred, y):
        m = y.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.log(y_pred[np.arange(m), y]))
    
    def feed_forward(self, x):
        self.z1 = x @ self.w1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = self.relu(self.z3)

        self.z4 = self.a3 @ self.w4 + self.b4
        self.a4 = self.relu(self.z4)

        self.z5 = self.a4 @ self.w5 + self.b5
        self.a5 = self.softmax(self.z5)

        return self.a5
    
    def backward(self, x, y):
        """Retropropagation du gradient et mise à jour des poids via momentum"""
        m = x.shape[0]
        beta = 0.9 # Coefficient du Momentum

        # Encodage one-hot des labels
        y_like_a5 = np.zeros_like(self.a5)
        y_like_a5[np.arange(m), y] = 1

        # CALCUL DES GRADIENTS 
        dz5 = self.a5 - y_like_a5
        dw5 = (self.a4.T @ dz5) / m
        db5 = np.sum(dz5, axis=0)/m

        da4 =  dz5 @ self.w5.T
        dz4 = da4 * (self.z4 > 0)
        dw4 = (self.a3.T @ dz4) / m
        db4 = np.sum(dz4, axis=0) / m
        
        da3 = dz4 @ self.w4.T
        dz3 = da3 * (self.z3 > 0)
        dw3 = (self.a2.T @ dz3) / m
        db3 = np.sum(dz3, axis=0) / m

        da2 = dz3 @ self.w3.T
        dz2 = da2 * (self.z2 > 0)
        dw2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0) / m

        da1 = dz2 @ self.w2.T
        dz1 = da1 * (self.z1 > 0)
        dw1 = (x.T @ dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        # MISE À JOUR DES PARAMÈTRES (Optimiseur Momentum) 
        # Formule : v = beta * v + (1 - beta) * gradient
        # param = param - lr * v

        self.v_w1 = beta * self.v_w1 + (1 - beta) * dw1
        self.v_b1 = beta * self.v_b1 + (1 - beta) * db1
        self.w1 -= self.lr * self.v_w1
        self.b1 -= self.lr * self.v_b1

        self.v_w2 = beta * self.v_w2 + (1 - beta) * dw2
        self.v_b2 = beta * self.v_b2 + (1 - beta) * db2
        self.w2 -= self.lr * self.v_w2
        self.b2 -= self.lr * self.v_b2

        self.v_w3 = beta * self.v_w3 + (1 - beta) * dw3
        self.v_b3 = beta * self.v_b3 + (1 - beta) * db3
        self.w3 -= self.lr * self.v_w3
        self.b3 -= self.lr * self.v_b3

        self.v_w4 = beta * self.v_w4 + (1 - beta) * dw4
        self.v_b4 = beta * self.v_b4 + (1 - beta) * db4
        self.w4 -= self.lr * self.v_w4
        self.b4 -= self.lr * self.v_b4

        self.v_w5 = beta * self.v_w5 + (1 - beta) * dw5
        self.v_b5 = beta * self.v_b5 + (1 - beta) * db5
        self.w5 -= self.lr * self.v_w5
        self.b5 -= self.lr * self.v_b5

if __name__ == "__main__":
    
    # Chargement du dataset MNIST
    train_dataset = torchvision.datasets.MNIST(
        root="./data", 
        train=True, 
        download=False, #download = True, pour telecharger le dataset
        transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", 
        train=False, 
        download=False, #download = True, pour telecharger le dataset
        transform=transform
    )

    # Creation des DataLoaders pour iterer par mini-batch
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

    model = Model()

    for i in range(51):
        train_loss = 0
        for x, y in train_loader:
            # On aplatit l'image (28x28 -> 784)
            x_numpy = x.view(x.shape[0], -1).numpy()
            y_numpy = y.numpy()

            y_pred = model.feed_forward(x_numpy)
            train_loss += model.loss(y_pred, y_numpy)
            model.backward(x_numpy, y_numpy)
        
        avg_loss = train_loss / len(train_loader)
        if i % 10 == 0:
            print(f"Epoch {i}: loss {avg_loss.item():.4f}")

    # sauvegarde des poids entrainés 
    np.savez("pois_parametre.npz", 
             w1=model.w1, b1=model.b1, 
             w2=model.w2, b2=model.b2, 
             w3=model.w3, b3=model.b3, 
             w4=model.w4, b4=model.b4,
             w5=model.w5, b5=model.b5)

    print("Paramètres sauvegardés dans 'pois_parametre.npz'")   
