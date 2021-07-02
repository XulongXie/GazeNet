The first thing that needs to be explained is the use process of the entire project file:
1. Data collection
2. Data expansion
3. Data annotation
4. Model training
5. Prediction and verification

=============================================================================================================
1. Data collection

You need the following tools:
1. Compilation environment: It is recommended to use pycharm IDE to compile under window 10.

2. Environment: It is recommended to use Anaconda to manage an environment, if you have a better choice.

3. Python function package: If it is the environment that comes with Anaconda, you need to check whether there are  
   numpy, opencv, tkinter and _thread, you may be missing an opencv at most, you can install it directly in Anaconda.

4. Before running, let’s look at the catchPhoto() function in dataCollection.py. The variable base_dir represents this  
   folder. You need to change it according to your own situation. Just modify the path to the corresponding one.

5. Then start running the program, just run the GUI directly, wait until the camera pop-up window opens and click the  
   yellow button to start, and then click some camera pop-up windows (very important), then you can see that a red dot will appear on your computer screen. What you need to do is to place your head as much as possible in the area inside the red box in the pop-up window, sitting correctly but also relaxing. You need to focus on the red dots on your screen. You can choose not to wear glasses. If you wear glasses, try to bring them better. It’s very simple. You need to press the Enter button to take a photo, which means, You only need to take a photo. At this time, you can base on the rules of MyGaze A or MyGaze B. Note that because there are two threads, the camera window may get stuck after taking a picture, which is normal, so don’t press the Enter key quickly, just wait for the window video stream to resume smoothly and press Enter again.

6. There are 9 points in total, and it will take you about 10 minutes in total.

7. Summary:
          1. Modify the path in the catchPhoto function in GUI.py, be careful not to use the Chinese path, or you can 
             change it to a relative path
          2. Run GdataCollectionUI.py
          3. Click the yellow button while waiting for the pop-up window
          4. Click again on the pop-up camera window (important!!! Otherwise it will cause the photo to fail)
          5. Follow the suggestion in the thesis when you change your head posture
          6. Every time you change your posture, press Enter to take a picture, and wait for the video stream in the 
             camera window to take a picture when it doesn’t freeze.
          7. Until all the points are taken

8. Tips:
             1. If possible, try to place the computer (if it is a laptop) as high as your head, you can choose to pad a
                few books
             2. If you want to change the head posture by raising your head and lower your head, don’t shift to the left 
                or right too much, just a little bit

=============================================================================================================
2. Data expansion
After the data collection completed, you have to expand your dataset so that you can get more samples for your own created dataset, everthing was done properly in this project, so you only need to do is run the expandData.py, then you can see a new folder be created, just click and check if everything is correct.

Then run copy.py or multi_copy.py in the utils folder, which depends on which model you want to use, after running this programm, you can find there are many newly created sample in your latest dataset, what you should do is just just keep them like this.

=============================================================================================================
3. Data annotation

For data annotation you can just run label.py, so that you can get your labels in "label" folder, after running it, just go and check if it's correct. Now in order to save the resource, you can just delete the folder which only contains the expanded data.

=============================================================================================================
4. Model training

The next steop is to train the model, the models you can find in the folder "models", everthing was finished with PyTorch, so you have to install it. Go to the models folder and run train.py, you can change some parameters there such as training epoch, and just wait until it finish.
One thing should be attention: you have to download shape_predictor_68_face_landmarks.dat file by youself, you can find it here: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
=============================================================================================================
5. Prediction and verification
After training, you can run evaluation.py to check if the result is good enough or if it's over-fitted, for GazeTracker1 structured model, you can also find pridict.py and video.py which allows you to double check the result.

=============================================================================================================
6. Others

1. All the functions are finished properly, this project contains two models: GazeTracker1 and GazeTracker2, which is named as "GazeNet_model" and "GazeNet_end2end" respectively, so you can choose which you want to use, just recommand that the "GazeNet_model" might be more suitable for you. The functions in these tuo folders are similar in general.
2. The pre-trained weights are only for GazeTracker1 available.