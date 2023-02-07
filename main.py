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
import pandas as pd
import pickle
import joblib
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
#Food Vision Model
model_food_vision_dir = "food_vision_model.h5"
model_food_vision = load_model(model_food_vision_dir,
                               custom_objects={'KerasLayer':hub.KerasLayer})
class_predictions_food_vision = array(['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon' , 'hamburger', 'ice_cream' , 'pizza', 'ramen' , 'steak', 'sushi'])

#Indian Food Vision Model
model_indian_food_vision_dir = "indian_food_vision_model.h5"
model_indian_food_vision = load_model(model_indian_food_vision_dir)
class_predictions_indian_food_vision = array(['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa'])

#Fruits Vision Model
model_fruit_vision_dir = "fruits_vision_model.h5"
model_fruit_vision = load_model(model_fruit_vision_dir)
class_predictions_fruit_vision = array(["Apple Golden 1","Avocado","Banana","Cherry 1","Cocos","Kiwi",
         "Lemon","Mango","Orange"])

#Sign Language Model
model_sign_language_dir = "sign_language_model.h5"
model_sign_language = load_model(model_sign_language_dir)
class_predictions_sign_language = array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Delete', 'Nothing','Space'])

#Car Price Model
model_used_car_price_dir = "used_car_prediction_model.joblib"
model_used_car_price = joblib.load(model_used_car_price_dir)



@app.get("/")
async def root():
    return {"message": "Welcome to the Food Vision API!"}


@app.get("/prediction/food-vision/")
async def get_image_prediction_food_vision(image_link: str = ""):
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
        "model_prediction": pred_class,
        "model_prediction_confidence_score": model_score
    }


@app.get("/prediction/indian-food-vision/")
async def get_image_prediction_indian_food_vision(image_link: str = ""):
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
        "model_prediction": pred_class,
        "model_prediction_confidence_score": model_score
    }

@app.get("/prediction/fruit-vision/")
async def get_image_prediction_fruit_vision(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}

    img_path = get_file(
        origin=image_link
    )
    img = load_img(
        img_path,
        target_size=(100, 100)
    )

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)
    img_array/=255.

    pred = model_fruit_vision.predict(img_array)
    score = softmax(pred[0])
    index = argmax(pred)

    pred_class = str(class_predictions_fruit_vision[index].title())

    model_score = round(max(score) * 100, 2)

    return {
        "model_prediction": pred_class,
        "model_prediction_confidence_score": model_score
    }

@app.get("/prediction/sign-language/")
async def get_image_prediction_sign_language(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}

    img_path = get_file(
        origin=image_link
    )
    img = load_img(
        img_path,
        target_size=(64, 64)
    )

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)
    #img_array/=255.

    pred = model_sign_language.predict(img_array)
    score = softmax(pred[0])
    index = argmax(pred)

    pred_class = str(class_predictions_sign_language[index].title())

    model_score = round(max(score) * 100, 2)

    return {
        "model_prediction": pred_class,
        "model_prediction_confidence_score": model_score
    }
@app.get("/prediction/used-car-price/")
async def get_prediction_used_car_price(location: str = "", year: str = "", km_driven: str = "", fuel: str = "", owners: str = "", transmission: str = "", seats: str = "",mileage: str = "", engine: str = "", power: str = ""):
    #if any parameter is missing, return an error message
    if location == "" or year == "" or km_driven == "" or fuel == "" or owners == "" or transmission == "" or seats == "" or mileage == "" or engine == "" or power == "":
        return {"message": "Please provide all the parameters"}

    def convert_var(location, owner, fuel, transmission):
        Location_Bangalore = 0
        Location_Chennai = 0
        Location_Coimbatore = 0
        Location_Delhi = 0
        Location_Hyderabad = 0
        Location_Jaipur = 0
        Location_Kochi = 0
        Location_Kolkata = 0
        Location_Mumbai = 0
        Location_Pune = 0
        Fuel_Type_Diesel = 0
        Fuel_Type_LPG = 0
        Fuel_Type_Petrol = 0
        Transmission_Manual = 0

        loc = ('Location_' + location)
        vars()[loc] = 1
        fu = ('Fuel_Type_' + fuel)
        vars()[fu] = 1
        if(transmission == 'Manual'):
            tran = ('Transmission_' + transmission)
            vars()[tran] = 1


        if owner == 'First':
            owner = 1
        elif owner == 'Second':
            owner = 2
        elif owner == 'Third':
            owner = 3
        else:
            owner = 4

        return ( Location_Bangalore, Location_Chennai, Location_Coimbatore, Location_Delhi,
                Location_Hyderabad, Location_Jaipur, Location_Kochi, Location_Kolkata, Location_Mumbai,
                Location_Pune, Fuel_Type_Diesel, Fuel_Type_LPG, Fuel_Type_Petrol,
                Transmission_Manual, owner)

    #convert the parameters to the correct format

    Location_Bangalore, Location_Chennai, Location_Coimbatore, Location_Delhi, \
    Location_Hyderabad, Location_Jaipur, Location_Kochi, Location_Kolkata, Location_Mumbai, Location_Pune, \
     Fuel_Type_Diesel, Fuel_Type_LPG, Fuel_Type_Petrol, \
    Transmission_Manual, owner = convert_var(location, owners, fuel, transmission)

    inputs = [[year, km_driven, owner, seats, mileage, engine, power,
               Location_Bangalore, Location_Chennai, Location_Coimbatore,
               Location_Delhi, Location_Hyderabad, Location_Jaipur,
               Location_Kochi, Location_Kolkata, Location_Mumbai, Location_Pune,
               Fuel_Type_Diesel, Fuel_Type_LPG, Fuel_Type_Petrol, Transmission_Manual]]


    #array = {'Year':[2015],'Kilometers_Driven':[41000 ],'Owner_Type':[1],'Seats':[5],'Mileage(km/kg)':[19.66],'Engine(CC)':[1582.0],'Power(bhp)':[126.20],'Location_Bangalore':[0],'Location_Chennai':[0],'Location_Coimbatore':[0],'Location_Delhi':[0],'Location_Hyderabad':[0],'Location_Jaipur':[0],'Location_Kochi':[0],'Location_Kolkata':[0],'Location_Mumbai':[0],'Location_Pune':[1],'Fuel_Type_Diesel':[1],'Fuel_Type_LPG':[0],'Fuel_Type_Petrol':[0],'Transmission_Manual':[1]}
    #df = pd.DataFrame(array)
    #pred = model_used_car_price.predict(df)
    #score = softmax(pred)
    #pred_class = str(pred)

    pred = "{:.2f}".format(model_used_car_price.predict(inputs)[0] * 100000)
    #score = softmax(pred)
    #model_score = round(max(score) * 100, 2)

    return pred




if __name__ == "__main__":
    uvicorn.run(app,port=int(os.environ.get('PORT', 8080)), host="0.0.0.0")


#Behind The Codes