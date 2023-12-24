# app.py
import streamlit as st
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from scipy import stats
import tensorflow as tf


names = ["P", "V"]
usernames = ["p", "v"]


file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "Classification", "abcdf", cookie_expiry_days=2)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Incorrect Username or Password")

if authentication_status == None:
    st.warning("Please enter Username and Password")

if authentication_status:

    model_path = 'Project/model_vgg19_2'

    model = tf.keras.models.load_model(model_path)

    csv_model = tf.keras.models.load_model('csv_model')

    window_size = 1000
    maximum_counting = 10000



    def classify_beat(input_image):


        if input_image is not None:
            pil_image = Image.open(input_image).convert("RGB").resize((224, 224))

            st.image(pil_image, caption="Uploaded Image.", use_column_width=True)

            x2 = image.img_to_array(pil_image)
            x2 = np.expand_dims(x2, axis=0)
            img_data2 = preprocess_input(x2)
            result2 = model.predict(img_data2)
            itemindex2 = np.argmax(result2, axis=1)
            itemindex2 = np.where(result2 == np.max(result2))
            reverse_mapping = ['F', 'S', 'V', 'Q', 'N', 'M']
            prediction_name = reverse_mapping[itemindex2[1][0]]

        
            prediction_mapping = {
        'F': '''Fusion of ventricular and normal beat.
            Fusion of ventricular and normal beat occurs when a normal beat and a ventricular contraction fuse together. 
            This can be observed in the electrocardiogram (ECG) as a combined waveform characteristic of both normal and ventricular beats.''',

        'S': '''Supraventricular premature beat.
            Supraventricular premature beat refers to an early heartbeat originating above the heart's ventricles. 
            This can result from abnormal electrical activity in the atria, leading to a premature contraction before the expected time.''',

        'V': '''Premature ventricular contraction.
            Premature ventricular contraction is an early heartbeat originating in the ventricles. 
            In this case, the lower chambers of the heart (ventricles) contract prematurely, disrupting the normal heart rhythm.''',

        'Q': '''Unclassifiable beat.
            Unclassifiable beat refers to a heartbeat that cannot be categorized based on the available classes. 
            This may occur when the ECG signal exhibits patterns or characteristics that do not fit into any predefined category.''',

        'N': '''Normal beat.
            A normal beat is a regular heartbeat that occurs at the expected time. 
            This is the standard and expected pattern of heart activity, representing a normal and healthy cardiac rhythm.''',

        'M': '''Myocardial infarction.
            Myocardial infarction is a heart attack that can be indicated by specific patterns in the ECG. 
            These patterns may include ST-segment elevation or depression, helping to diagnose the presence of a heart attack.''',
                }

            prediction_description = prediction_mapping.get(prediction_name, 'Unknown')

            st.write(f"The predicted type is **{prediction_description}**.")

    def classify_csv(record_file, annotation_file):

            signals = []

            with record_file:
                df = pd.read_csv(record_file)

                signals = df.iloc[:, 1].tolist()

            signals = stats.zscore(signals)

            X = []

            with annotation_file:
                data = annotation_file.read().decode("utf-8").splitlines()
                for d in range(1, len(data)): 
                    splitted = data[d].split(' ')
                    splitted = filter(None, splitted)
                    next(splitted)  
                    pos = int(next(splitted)) 

                    if window_size <= pos and pos < (len(signals) - window_size):
                        beat = signals[pos - window_size:pos + window_size] 
                        X.append(beat)


            df_X = pd.DataFrame(X)

            pred = csv_model.predict(df_X)

            classes = ['N', 'L', 'R', 'A', 'V']
            predicted_class = classes[np.argmax(pred)]

            prediction_mapping = {
        'N': '''Normal beat.
            The 'N' represents a normal heartbeat or cardiac complex. It indicates that the electrical activity of the heart is within the normal range, and there are no abnormalities or irregularities in the heart rhythm. 
            This is the expected and healthy pattern of heart activity.''',

        'L': '''Left bundle branch block.
            An 'L' complex indicates the presence of a left bundle branch block. This is an abnormality in the electrical conduction system of the heart, where the electrical signals do not travel normally through the left bundle branch. 
            Left bundle branch block can affect the timing and coordination of ventricular contractions, leading to an altered ECG pattern.''',

        'R': '''Right bundle branch block.
            An 'R' complex indicates a right bundle branch block. Similar to left bundle branch block, this signifies an abnormality in the electrical conduction system involving the right bundle branch. 
            Right bundle branch block can also impact the timing and coordination of ventricular contractions, resulting in characteristic changes in the ECG.''',

        'A': '''Atrial premature beat.
            An 'A' complex represents an atrial premature beat, which is an early contraction originating in the atria (upper chambers of the heart) before the next expected normal heartbeat. 
            This premature beat can disrupt the regular heart rhythm and is often identified by distinctive features in the ECG waveform.''',

        'V': '''Ventricular premature beat.
            A 'V' complex represents a ventricular premature beat, which is an early contraction originating in the ventricles (lower chambers of the heart) before the next expected normal heartbeat. 
            This premature beat arises from the ventricles and can cause irregularities in the ECG pattern, indicating potential issues with the heart's electrical conduction system.''',
                }



            prediction_description = prediction_mapping.get(predicted_class, 'Unknown')
            st.write(f"The predicted type is **{prediction_description}**.")





    def main():
        authenticator.logout("Logout", "sidebar")
        st.sidebar.title(f"Arrhythmia Classification | Welcome {name}")
        selected_option = st.sidebar.radio("Choose an option:", ["Image Classification", "CSV Classification"])

        if selected_option == "Image Classification":
        
            st.title("Arrhythmia Classification with Image")
            input_image = st.file_uploader("Choose an ECG image...", type="png")
            if input_image is not None:
                classify_beat(input_image)

        elif selected_option == "CSV Classification":
            st.title("CSV File Prediction")

            record_file = st.file_uploader("Upload CSV Record File", type=["csv"])
            annotation_file = st.file_uploader("Upload Annotation File", type=["txt"])

            if record_file is not None and annotation_file is not None:
                classify_csv(record_file, annotation_file)


    if __name__ == "__main__":
        main()
