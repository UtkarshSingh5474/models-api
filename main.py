from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims
from tensorflow.nn import softmax
from numpy import argmax
from numpy import max
from numpy import array
import uvicorn
import os

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

model_food_vision_dir = "food_vision_model.h5"
model_food_vision = load_model(model_food_vision_dir,
                               custom_objects={'KerasLayer':hub.KerasLayer})

class_predictions_food_vision = array(['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon' , 'hamburger', 'ice_cream' , 'pizza', 'ramen' , 'steak', 'sushi'])

model_indian_food_vision_dir = "indian_food_vision_model.h5"
model_indian_food_vision = load_model(model_indian_food_vision_dir)

class_predictions_indian_food_vision = array(['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa'])


@app.get("/")
async def root():
    return {"message": "Welcome to the Food Vision API!"}


@app.get("/net/image/prediction/food-vision/")
async def get_net_image_prediction_food_vision(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}

    img_path = get_file(
        origin=image_link
    )
    img = load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)

    # Rescale the image (get all values between 0 and 1)
    img = img_array / 255.


    pred = model_food_vision.predict(img)
    score = softmax(pred[0])


    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
        pred_class = class_predictions_food_vision[pred.argmax()]  # if more than one output, take the max
    else:
        pred_class = class_predictions_food_vision[int(round(pred)[0][0])]  # if only one output, round

    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": pred_class,
        "model-prediction-confidence-score": model_score
    }


@app.get("/net/image/prediction/indian-food-vision/")
async def get_net_image_prediction_indian_food_vision(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}

    img_path = get_file(
        origin=image_link
    )
    img = load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)
    img_array/=255.

    pred = model_indian_food_vision.predict(img_array)
    score = softmax(pred[0])
    index = argmax(pred)

    pred_class = str(class_predictions_indian_food_vision[index].title())

    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": pred_class,
        "model-prediction-confidence-score": model_score
    }


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app,port=port, host="0.0.0.0")
