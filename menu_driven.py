import cv2
import glob
import os
import numpy as np
import dlib


# Global variables:
video_path="video\\"
img_path="images\\"
norm_img_path="normalised_images\\"
# video_path="F:\\BE_Project\\video\\"
# img_path="F:\\BE_Project\\images\\"
# norm_img_path="F:\\BE_Project\\normalised_images\\"
faceDet=cv2.CascadeClassifier('faceclassifier\\haarcascade_frontalface_alt.xml')
faceDet_two=cv2.CascadeClassifier('faceclassifier\\haarcascade_frontalface_alt_tree.xml')
faceDet_three=cv2.CascadeClassifier('faceclassifier\\haarcascade_frontalface_alt2.xml')
faceDet_four=cv2.CascadeClassifier('faceclassifier\\haarcascade_frontalface_default.xml')
data={}

def start_recording(path,name):
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path+name+".avi",fourcc, 20.0, (640,480))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,1)
            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    print("Stored successfully")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def split_video(path,name):
    # print(path)
    vidObj=cv2.VideoCapture(path)

    count=0
    success=1

    while success:
        vidObj.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        success,image=vidObj.read()
        img_last=cv2.imread(name+"{}.jpg".format(count-1))
        if np.array_equal(image,img_last):
            break

        cv2.imwrite(img_path+name+'%d.jpg'%count, image)
        # print("success")
        count+=1
    print("Extracted %d frames successfully to "%count,img_path+name)

def extract_features(path,name):
    print("Please wait while features are being extracted!")
    files=glob.glob('images\\'+name+'*')
    filenumber=0
    for f in files:
        frame=cv2.imread(f) #open image
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #conver to grayscale

        # Detect faces using 4 different classfiers
        face=faceDet.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(5,5),flags=cv2.CASCADE_SCALE_IMAGE)
        face_two=faceDet_two.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(5,5),flags=cv2.CASCADE_SCALE_IMAGE)
        face_three=faceDet_three.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(5,5),flags=cv2.CASCADE_SCALE_IMAGE)
        face_four=faceDet_four.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(5,5),flags=cv2.CASCADE_SCALE_IMAGE)

        # Go over detected faces, stop at first detected face, return empty if no face found
        if len(face)==1:
            faceFeatures=face
        elif len(face_two)==1:
            faceFeatures=face_two
        elif len(face_three)==1:
            faceFeatures=face_three
        elif len(face_four)==1:
            faceFeatures=face_four
        else:
            faceFeatures=""
            
        # Cut and save face
        for(x,y,w,h) in faceFeatures: #get coordinates and size of rectangle containing face
            # print("face found inf file: %s %f")
            gray=gray[y:y+h,x:x+w] #cut frame to size
            try:
                out=cv2.resize(gray,(350,350))
                cv2.imwrite(norm_img_path+name+'%s.jpg'%(filenumber),out)
            except:
                pass
        filenumber+=1
    print("Successfully extracted features")

def get_files(imgList,name):
    checking=glob.glob(norm_img_path+name+'*.jpg')
    prediction=checking[:int(len(checking))]

    return prediction

def get_landmarks(image):
    # import dlib
    import math

    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        
        # for a in range(1,68):
        #     print(xlist[a],ylist[a])

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
        # print("landmark found")
        
    if len(detections) < 1:
        data['landmarks_vectorised'] = "error"

def make_sets(name,clahe):
    prediction_data=[]
    prediction_labels=[]
    imgList=["none","mild","severe"]
    
    for img in imgList:
        prediction=get_files(imgList,name)
        
        for item in prediction:
            image=cv2.imread(item)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            clahe_image=clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised']=="error":
                pass
                # print("no face detected on this while prediction")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(imgList.index(img))
    return prediction_data,prediction_labels

def predict(path,name):
    from sklearn.svm import SVC
    import _pickle as pickle
    from collections import Counter

    print("Result is being calculated wait for the moment")

    clf = SVC(kernel='linear', probability=True, tol=1e-3) #,verbose = True) #Set the classifier as a support vector machines with polynomial kernel
    img_pickle_path='trained_model.pkl'
    img_unpickle=open(img_pickle_path,'rb')
    clf=pickle.load(img_unpickle)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    prediction_data, prediction_lables=make_sets(name,clahe)
    npar_pred=np.array(prediction_data)
    # npar_predlab=np.array(prediction_lables)
    # pred_lin=clf.score(npar_pred,prediction_lables)
    predicted=clf.predict(npar_pred)
    predicted=list(predicted)
    most_common,num_most_common=Counter(predicted).most_common(1)[0]
    # print(most_common,num_most_common)
    return most_common

if __name__ == "__main__":
    os.system('cls')
    name=input("Enter Name: ")
    if(os.path.isfile(video_path+name+'.avi')):
        print("Welcome ",name)
        
        print("\n Answer the following questions with the options: \n1. Not at all \n2.Several days \n3. More than half the days \n4. Nearly everyday \n ")
        num1=int(input("\n1. Little interest or pleasure in doing thing: "))
        num2=int(input("\n2. Feeling down, depressed, or hopeless: "))
        num3=int(input("\n3. Trouble falling or staying asleep, or sleeping too much: "))
        num4=int(input("\n4. Feeling tired or having little energy: "))
        num5=int(input("\n5. Poor appetite or overeating: "))
        num6=int(input("\n6. Feeling bad about yourself, or that you are a failure, or have let yourself or your family down: "))
        num7=int(input("\n7. Trouble concentrating on things, such as reading the newspaper or watching television: "))
        num8=int(input("\n8. Moving or speaking so slowly that other people could have noticed? Or the opposite, being so fidgety or restless that you have been moving around a lot more than usual: "))
        num9=int(input("\n9. Thoughts that you would be better off dead or of hurting yourself in some way: "))
        phq_score=num1+num2+num3+num4+num5+num6+num7+num8+num9

        print("Please be patient while programming is running ...")

        if(phq_score):
            split_video(video_path+name+'.avi',name)
            extract_features(img_path+name,name)
            predicted=predict(norm_img_path+name,name)

            if(phq_score>=1 or phq_score<=12):
                if(predicted==0):
                    final_result=7.2+0.4*phq_score
                if(predicted==1):
                    final_result=14.4+0.4*phq_score
                if(predicted==2):
                    final_result=21.6+0.4*phq_score
                

            elif(phq_score>=13 or phq_score<=24):
                if(predicted==0):
                    final_result=7.2+0.4*phq_score
                if(predicted==1):
                    final_result=14.4+0.4*phq_score
                if(predicted==2):
                    final_result=21.6+0.4*phq_score

            else:
                if(predicted==0):
                    final_result=7.2+0.4*phq_score
                if(predicted==1):
                    final_result=14.4+0.4*phq_score
                if(predicted==2):
                    final_result=21.6+0.4*phq_score

            if(final_result>=1 and final_result<=12):
                print("Not depressed")
            elif(final_result>=13 and final_result<=24):
                print("Mildly depressed")
            else:
                print("Severely depressed visit a doctor asap")

        

        # if(phq_score):
        #     choice=int(input("Enter choice: \n1. Split video \n2. Extract Features \n3. Predict \n4. Exit"))
        #     while(choice):
        #         if(choice==1):
        #             split_video(video_path+name+'.avi',name)
        #             break
                
        #         if(choice==2):
        #             extract_features(img_path+name,name)
        #             break

        #         if(choice==3):
        #             predict(norm_img_path+name,name)
        #             break
                
        #         if(choice==4):
        #             exit()
        # else:
        #     pass
    else:
        start_recording(video_path+name+'.avi',name)
    