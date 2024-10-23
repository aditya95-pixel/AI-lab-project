import os
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    


# loading the saved models

diabetes_model = pickle.load(open(f'diabetes_model.pkl', 'rb'))
diabetes_scaler=pickle.load(open(f'diabetes_scaler.pkl', 'rb'))
heart_disease_model = pickle.load(open(f'heart_disease_model.pkl', 'rb'))
heart_disease_scaler = pickle.load(open(f'heart_scaler.pkl', 'rb'))
parkinsons_model = pickle.load(open(f'parkinsons_model.pkl', 'rb'))
parkinsons_scaler = pickle.load(open(f'parkinsons_scaler.pkl', 'rb'))
lung_model = pickle.load(open(f'lung_cancer_model.pkl', 'rb'))
lung_scaler = pickle.load(open(f'lung_cancer_scaler.pkl', 'rb'))
kidney_model=pickle.load(open(f'kidney_model.pkl', 'rb'))
kidney_scaler=pickle.load(open(f'kidney_scaler.pkl','rb'))
liver_model=pickle.load(open(f'liver_model.pkl', 'rb'))
liver_scaler=pickle.load(open(f'liver_scaler.pkl','rb'))
# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Lung Cancer Prediction',
                            'Chronic Kidney Disease Prediction',
                            'Liver Disease Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person','person','heart','activity'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]
        user_input=np.asarray(user_input)
        user_input=user_input.reshape(1,-1)
        scaled_features = diabetes_scaler.transform(user_input)
        diab_prediction = diabetes_model.predict(scaled_features)

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex male:1 female:0')

    with col3:
        cp = st.text_input('Chest Pain types 1:typical angina 2:atypical angina 3:non-anginal pain 4:asymptomatic')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure mm/Hg')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl 1:true 0:false')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results 0:normal 1:abnormality (Stress Test elevation or depression of >0.05mV) 2:Prbobale or definite left ventricular hypertrophy by Estes\' criteria')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina 1:yes 0:no')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment 1:up sloping 2:flat 3:down sloping')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy (0-3)')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        user_input=np.asarray(user_input)
        user_input=user_input.reshape(1,-1)
        scaled_features = heart_disease_scaler.transform(user_input)
        heart_prediction = heart_disease_model.predict(scaled_features)

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]
        user_input=np.asarray(user_input)
        user_input=user_input.reshape(1,-1)
        scaled_features = parkinsons_scaler.transform(user_input)
        parkinsons_prediction = parkinsons_model.predict(scaled_features)
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

#lung cancer prediction
if selected == 'Lung Cancer Prediction':

    # page title
    st.title('Lung Cancer Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        gender= st.text_input('Gender 0:male 1:female')

    with col2:
        age= st.text_input('Age')

    with col3:
        smoke= st.text_input('Smoke 2:YES 1:NO')

    with col1:
        yellow= st.text_input('Yellow fingers 2:YES 1:NO')

    with col2:
        anxiety = st.text_input('Anxiety 2:YES 1:NO')

    with col3:
        peer= st.text_input('Peer Pressure 2:YES 1:NO')

    with col1:
        chronic= st.text_input('Chronic Disease 2:YES 1:NO')

    with col2:
        fatigue= st.text_input('Fatigue 2:YES 1:NO')

    with col3:
        allergy= st.text_input('Allergy 2:YES 1:NO')

    with col1:
        wheeze= st.text_input('Wheezing 2:YES 1:NO')
    
    with col2:
        alcohol= st.text_input('Alcohol 2:YES 1:NO')

    with col3:
        cough= st.text_input('Cough 2:YES 1:NO')
    
    with col1:
        short= st.text_input('Shortness of breath 2:YES 1:NO')

    with col2:
        swallow= st.text_input('Swallowing difficulty 2:YES 1:NO')

    with col3:
        cp= st.text_input('Chest Pain 2:YES 1:NO')

    # code for Prediction
    lung_diagnosis = ''

    # creating a button for Prediction

    if st.button('Lung Cancer Test Result'):

        user_input = [gender,age,smoke,yellow,anxiety,peer,chronic,fatigue,allergy,wheeze,alcohol,cough,short,swallow,cp]

        user_input = [float(x) for x in user_input]
        user_input=np.asarray(user_input)
        user_input=user_input.reshape(1,-1)
        scaled_features = lung_scaler.transform(user_input)
        lung_prediction = lung_model.predict(scaled_features)

        if lung_prediction[0] == 1:
            lung_diagnosis = 'The person has lung cancer'
        else:
            lung_diagnosis = 'The person does not have lung cancer'

    st.success(lung_diagnosis)

#chronic kidney disease prediction
if selected == 'Chronic Kidney Disease Prediction':

    # page title
    st.title('Chronic Kidney Disease Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        bp = st.text_input('Blood Pressure (bp in mm/Hg)')

    with col3:
        sg = st.text_input('Specific Gravity (sg in range 1.005 to 1.025)')

    with col1:
        al = st.text_input('Albumin (al: 0 to 5)')

    with col2:
        su = st.text_input('Sugar (su: 0 to 5)')

    with col3:
        rbc = st.text_input('Red Blood Cells (rbc: 1=normal, 0=abnormal)')

    with col1:
        pc = st.text_input('Pus Cell (pc: 1=normal, 0=abnormal)')

    with col2:
        pcc = st.text_input('Pus Cell Clumps (pcc: 1=present, 0=not present)')

    with col3:
        ba = st.text_input('Bacteria (ba: 1=present, 0=not present)')

    with col1:
        bgr = st.text_input('Blood Glucose Random (bgr in mgs/dl)')

    with col2:
        bu = st.text_input('Blood Urea (bu in mgs/dl)')

    with col3:
        sc = st.text_input('Serum Creatinine (sc in mgs/dl)')

    with col1:
        sod = st.text_input('Sodium (sod in mEq/L)')

    with col2:
        pot = st.text_input('Potassium (pot in mEq/L)')

    with col3:
        hemo = st.text_input('Hemoglobin (hemo in gms)')

    with col1:
        pcv = st.text_input('Packed Cell Volume (pcv)')

    with col2:
        wc = st.text_input('White Blood Cell Count (wc in cells/cumm)')

    with col3:
        rc = st.text_input('Red Blood Cell Count (rc in millions/cmm)')

    with col1:
        htn = st.text_input('Hypertension (htn: 1=Yes, 0=No)')

    with col2:
        dm = st.text_input('Diabetes Mellitus (dm: 1=Yes, 0=No)')

    with col3:
        cad = st.text_input('Coronary Artery Disease (cad: 1=Yes, 0=No)')

    with col1:
        appet = st.text_input('Appetite (appet: 1=Good, 0=Poor)')

    with col2:
        pe = st.text_input('Pedal Edema (pe: 1=Yes, 0=No)')

    with col3:
        ane = st.text_input('Anemia (ane: 1=Yes, 0=No)')

    # code for Prediction
    kidney_diagnosis = ''

    # creating a button for Prediction
    if st.button('CKD Test Result'):

        # Prepare user input
        user_input = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]
        user_input = [float(x) for x in user_input]
        user_input = np.asarray(user_input).reshape(1, -1)

        # Feature scaling (assuming 'kidney_scaler' is your scaler)
        scaled_features = kidney_scaler.transform(user_input)

        # Prediction (assuming 'kidney_model' is your trained ML model)
        kidney_prediction = kidney_model.predict(scaled_features)

        if kidney_prediction[0] == 1:
            kidney_diagnosis = 'The person has chronic kidney disease'
        else:
            kidney_diagnosis = 'The person does not have chronic kidney disease'

    st.success(kidney_diagnosis)


if selected == 'Liver Disease Prediction':

    # Page title
    st.title('Liver Disease Prediction using ML')

    # Get the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        gender = st.selectbox('Gender', options=['Male', 'Female'])
        gender = 0 if gender == 'Male' else 1

    with col3:
        tb = st.text_input('Total Bilirubin (TB)')

    with col1:
        db = st.text_input('Direct Bilirubin (DB)')

    with col2:
        alkphos = st.text_input('Alkaline Phosphotase (Alkphos)')

    with col3:
        sgpt = st.text_input('Alamine Aminotransferase (Sgpt)')

    with col1:
        sgot = st.text_input('Aspartate Aminotransferase (Sgot)')

    with col2:
        tp = st.text_input('Total Proteins (TP)')

    with col3:
        alb = st.text_input('Albumin (ALB)')

    with col1:
        ag_ratio = st.text_input('Albumin and Globulin Ratio (A/G Ratio)')

    # Code for Prediction
    liver_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Liver Disease Test Result'):

        # Prepare the user input
        user_input = [age, gender, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]

        user_input = [float(x) for x in user_input]  # Convert inputs to float
        user_input = np.asarray(user_input).reshape(1, -1)

        # Feature scaling (assuming 'liver_scaler' is your scaler)
        scaled_features = liver_scaler.transform(user_input)

        # Prediction (assuming 'liver_model' is your trained ML model)
        liver_prediction = liver_model.predict(scaled_features)

        if liver_prediction[0] == 0:
            liver_diagnosis = 'The person has liver disease'
        else:
            liver_diagnosis = 'The person does not have liver disease'

    st.success(liver_diagnosis)
