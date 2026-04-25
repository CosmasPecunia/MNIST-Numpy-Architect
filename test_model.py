import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from main import Model 

#CHARGEMENT DU MODELE ET DES PARAMÈTRES
model = Model()
# On charge les paramètres 
params = np.load("pois_parametre.npz")
model.w1, model.b1 = params['w1'], params['b1']
model.w2, model.b2 = params['w2'], params['b2']
model.w3, model.b3 = params['w3'], params['b3']
model.w4, model.b4 = params['w4'], params['b4']
model.w5, model.b5 = params['w5'], params['b5']

print("Parametres chargés avec succès.")

# 2. INTERFACE GRAPHIQUE (TKINTER)

WIDTH, HEIGHT = 280, 280
root = tk.Tk()
root.title("Numpy MNIST Predictor")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black", cursor="pencil")
canvas.pack(pady=10)

# Image PIL pour stocker le dessin en arrière-plan
image_pil = Image.new("L", (WIDTH, HEIGHT), "black")
draw = ImageDraw.Draw(image_pil)

def draw_lines(event):
    r = 10  # Épaisseur du trait
    x, y = event.x, event.y
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse([x-r, y-r, x+r, y+r], fill="white")

canvas.bind("<B1-Motion>", draw_lines)

# 3. LOGIQUE DE PRÉDICTION
def predict():
    # 1. Préparation de l'image (28x28)
    img_resized = image_pil.resize((28, 28))
    img_array = np.array(img_resized) / 255.0  # Normalisation 0-1
    
    # 2. Aplatissement (Flatten) pour ton modèle NumPy
    img_final = img_array.reshape(1, 784)
    
    # 3. Passage dans le modèle (Inférence)
    output = model.feed_forward(img_final)
    
    # 4. Résultat
    prediction = np.argmax(output, axis=1)[0]
    confiance = np.max(output) * 100
    
    result_label.config(text=f"Prédiction : {prediction} ({confiance:.2f}%)")

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill="black")
    result_label.config(text="Dessine un chiffre")

#BOUTONS ET LABELS 
btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Prédire", command=predict, bg="#2ecc71", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5, pady=5)
tk.Button(btn_frame, text="Effacer", command=clear, bg="#e74c3c", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5, pady=5)

result_label = tk.Label(root, text="Dessine un chiffre", font=("Arial", 18))
result_label.pack(pady=20)

root.mainloop()