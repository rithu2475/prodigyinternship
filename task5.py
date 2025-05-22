import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Load MobileNetV2 pretrained model
model = MobileNetV2(weights='imagenet')
# Example: Simplified calorie mapping (can use a CSV from USDA too)
food_calories = {
    "pizza": 285,
    "cheeseburger": 303,
    "hotdog": 150,
    "apple": 95,
    "banana": 105,
    "french_fries": 312,
    "sushi": 200,
    "ice_cream": 137
}
def predict_food_and_calories(img_path):
    img = Image.open(img_path).resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]

    for food_name, description, confidence in decoded:
        food_item = food_name.lower().replace("_", " ")
        if food_item in food_calories:
            print(f"Food: {food_item.capitalize()}, Confidence: {confidence:.2f}")
            print(f"Estimated Calories: {food_calories[food_item]} kcal")
            return

    print("Food not found in calorie database. Try a different image.")
predict_food_and_calories("example_food.jpg")
