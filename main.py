from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from endpoint_auth.auth_func import get_current_user, create_access_token, authenticate_user
from datetime import timedelta,datetime
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
from secret.secret_keys import ACCESS_TOKEN_EXPIRE_MINUTES

from fastapi.responses import FileResponse, StreamingResponse
#import custom data types
from endpoint_auth.custom_data_types import User, AlphavtarInput

from connection.connect_db_user import convert_to_linux_path
import platform
# import sys
# import subprocess
from databases import Database
import os
import pandas as pd
import random
# import all other things
import cv2
import numpy as np
import os
import io
# Create a new database connection for the user database
if platform.system() == "Windows":
    database = Database('sqlite+aiosqlite:///db_user/users.db')
else:
    database = Database(convert_to_linux_path('sqlite+aiosqlite:///db_user/users.db'))
  

app = FastAPI(docs_url="/alphavtar/docs", redoc_url=None)


face_refer_list = ["a-Face (1).png","b-Face.png","b-Face-side.png","F-Face.png","G-Face(1).png","J-Face.png","o-Face.png","p-Face(1).png","Q-Face.png","R-Face-Side.png","P-Face.png","s-Face.png","T-Face(1).png","u-Face.png","A-FaceCounterUp.png","d-Face.png"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


################################################# FUNCTIONALITIES ###############################################

def create_face_combination(name, image_pool):
    max_parts = 7
    max_tries = 10
    attempt = 0

    while attempt < max_tries:
        try:
            image = {}
            name_list = list(name.lower())[:max_parts]
            print(f"Name List: {name_list}")

            # Select a face image
            face = random.choice(image_pool['Face'])
            face_letter = face.split("-")[0].lower()
            print(f"Selected Face: {face}, Face Letter: {face_letter}")

            if face_letter in name_list:
                name_list.remove(face_letter)
            image["Face"] = face

            other_elements = ["Lefteye", "Righteye", "Nose", "Mouth", "Leftear", "Rightear"]
            random.shuffle(other_elements)

            while name_list:
                if not other_elements:
                    raise IndexError("Ran out of elements to match remaining letters")

                part = other_elements.pop(0)
                potential_images = [img for img in image_pool[part] if img.split("-")[0][-1].lower() in name_list]

                if not potential_images:
                    continue  # Skip this part if no matching image is found

                to_add = random.choice(potential_images)
                part_letter = to_add.split("-")[0][-1].lower()

                if part_letter in name_list:
                    name_list.remove(part_letter)
                    image[part] = to_add
                    print(f"Selected {part}: {to_add}, Part Letter: {part_letter}, Remaining Name List: {name_list}")

                if not other_elements and name_list:
                    raise IndexError("Ran out of elements to match remaining letters")

            return image

        except IndexError:
            print("An error occurred. Retrying...")
            attempt += 1

    print("Max attempts reached. Unable to create face combination.")
    return "Not Possible"  # or handle the error as needed

#################################################################################################################
@app.on_event("startup")
async def database_connect():
    await database.connect()


@app.on_event("shutdown")
async def database_disconnect():
    await database.disconnect()

@app.get("/alphavtar")
def read_root():
    return {"Welcome to the Alphavtar API. This is the root endpoint."}

# for all random endpoints return 404
@app.get("/alphavtar/{random_endpoint}")
async def random_endpoint(random_endpoint: str):
    raise HTTPException(status_code=404, detail="Endpoint not found")

# Get a token for the user
@app.post("/alphavtar/v1/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}




@app.post("/alphavtar/v1/facegeneration")
async def alphavtar_input(alphavtar_input: AlphavtarInput, current_user: User = Depends(get_current_user)):
    dashboard_inputs = alphavtar_input.dict()
    name = dashboard_inputs['Name']
    name_list = [i.lower() for i in name]
    image_pool = {}
    face_list = []

    for file in os.listdir(r"Individual Elements"):
        if file.split("-")[0].lower() in name_list and "Face" in file:
            if file in face_refer_list:
                face_list.append(file)
    if len(face_list)==0:
        return "Face Not Possible for now :("
    image_pool["Face"] = face_list

    other_elements = ["Lefteye","Righteye","Nose","Mouth","Leftear","Rightear"]
    for i in other_elements:
        image_pool[i] = []
        for file in os.listdir(r"Contour"):
            if file.split("-")[0][-1].lower() in name_list and i in file and "brow" not in file:
                image_pool[i].append(file)
    
    result = create_face_combination(name, image_pool)
    if result=="Not Possible":
        return "Could not build image for given name :("
    for i in result:
        if i=="Face":
            result[i] = os.path.join(r"Individual Elements",result[i])
        else:
            result[i] = os.path.join(r"Contour",result[i])
    images = result

    # Read the face image with an alpha channel (if exists)
    img = cv2.imread(images["Face"], cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Error: Unable to read image at {images['Face']}")
        exit()

    # Check if the image has an alpha channel
    if img.shape[2] == 4:
        # Split the image into BGR and alpha channels
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]

        # Create a white background image
        white_background = np.ones_like(bgr, dtype=np.uint8) * 255

        # Blend the BGR image with the white background using the alpha channel
        alpha_normalized = alpha[:, :, np.newaxis] / 255.0
        img = bgr * alpha_normalized + white_background * (1 - alpha_normalized)
        img = img.astype(np.uint8)
    else:
        img = img

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to the image
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Variable to store the most circular contour
    most_circular_contour = None
    highest_circularity = 0

    # Loop through contours to find the most circular one
    for contour in contours:
        if len(contour) >= 5:  # Fit ellipse requires at least 5 points
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            (x, y), (MA, ma), angle = ellipse

            # Calculate circularity
            circularity = min(MA, ma) / max(MA, ma)
            
            # Check if this is the most circular contour found so far
            if circularity > highest_circularity:
                highest_circularity = circularity
                most_circular_contour = contour

    # Draw the bounding box around the most circular contour and label coordinates
    if most_circular_contour is not None:
        x, y, w, h = cv2.boundingRect(most_circular_contour)
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.putText(img, f"({x}, {y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Print the coordinates
        print(f"Coordinates: ({x}, {y})")
        print(f"Width and Height: ({w}, {h})")

        # Define relative positions and sizes of facial features
        features_possible = {
            'Lefteye': {'rel_x': 0.3, 'rel_y': 0.35, 'rel_w': 0.1, 'rel_h': 0.1},
            'Righteye': {'rel_x': 0.7, 'rel_y': 0.35, 'rel_w': 0.1, 'rel_h': 0.1},
            'Nose': {'rel_x': 0.5, 'rel_y': 0.55, 'rel_w': 0.15, 'rel_h': 0.15},
            'Mouth': {'rel_x': 0.5, 'rel_y': 0.75, 'rel_w': 0.2, 'rel_h': 0.1},
            'Leftear': {'rel_x': -0.05, 'rel_y': 0.5, 'rel_w': 0.1, 'rel_h': 0.2},
            'Rightear': {'rel_x': 1.05, 'rel_y': 0.5, 'rel_w': 0.1, 'rel_h': 0.2}
        }
        features = {}
        for i in images:
            if i in features_possible:
                features[i] = features_possible[i]
        

        for feature, values in features.items():
            fx = int(x + values['rel_x'] * w - (values['rel_w'] * w / 2))
            fy = int(y + values['rel_y'] * h - (values['rel_h'] * h / 2))
            fw = int(values['rel_w'] * w)
            fh = int(values['rel_h'] * h)

            # Read the feature image
            feature_img = cv2.imread(images[feature], cv2.IMREAD_UNCHANGED)

            if feature_img is not None:
                # Resize the feature image to the calculated size
                feature_img = cv2.resize(feature_img, (fw, fh), interpolation=cv2.INTER_AREA)

                # Check if the feature image has an alpha channel
                if feature_img.shape[2] == 4:
                    # Split the feature image into BGR and alpha channels
                    feature_bgr = feature_img[:, :, :3]
                    feature_alpha = feature_img[:, :, 3]

                    # Create a mask and inverse mask from the alpha channel
                    mask = cv2.merge([feature_alpha, feature_alpha, feature_alpha])
                    inv_mask = cv2.bitwise_not(mask)

                    # Region of interest (ROI) in the main image
                    roi = img[fy:fy+fh, fx:fx+fw]

                    # Black-out the area of the feature in the ROI
                    img_bg = cv2.bitwise_and(roi, inv_mask)

                    # Take only region of the feature from the feature image
                    feature_fg = cv2.bitwise_and(feature_bgr, mask)

                    # Add the feature to the ROI and modify the main image
                    dst = cv2.add(img_bg, feature_fg)
                    img[fy:fy+fh, fx:fx+fw] = dst

                    #cv2.putText(img, feature, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    print(f"{feature.capitalize()} - Coordinates: ({fx}, {fy}), Size: ({fw}, {fh})")
                else:
                    # If no alpha channel, just place the resized feature image
                    img[fy:fy+fh, fx:fx+fw] = feature_img

                    #cv2.putText(img, feature, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    print(f"{feature.capitalize()} - Coordinates: ({fx}, {fy}), Size: ({fw}, {fh})")
            else:
                print(f"Error: Unable to read image for {feature}")
    else:
        print("No circular contours found.")
    
    temp_file_path = f'temp_image.png'
    cv2.imwrite(temp_file_path, img)
    #return FileResponse(temp_file_path, media_type='image/png', filename='face_with_features.png')
    _, img_encoded = cv2.imencode('.png', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())
    return StreamingResponse(img_bytes, media_type='image/png')

        
@app.post("/alphavtar/v1/facedownload")
async def alphavtar_input(current_user: User = Depends(get_current_user)):
    image_path = "temp_image.png"
    return FileResponse(image_path, filename="file.png", media_type='application/octet-stream')