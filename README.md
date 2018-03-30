# Rock-paper-scissors-game-AI-Design
We proposed a Rock-paper-scissors game AI to play it with human, and it is able to beat human player every time. The gesture recognition is able to computed in real-time. Our algorithm includes four steps: hand detection, hand object tracking, hand gesture recognition and response of AI. The first step is to extract player’s hand object region. We used back projection to obtain the skin color region and Gaussian filter and Thresholding to remove noise. At the same time, we also need to track hand object position. Player can move their hand around the frame. We applied Meanshift algorithm to implement it. We also defined initial step and playing step. When the players are moving their hands. it is in a initial step. When the players are not moving hands, it is in a playing step. We detect the movement by the displacement of hand object and the variance of the hand in the bounding box. Finally, we applied Convolutional Neural Network Model to determine players’ hand gestures. We have diverse training data including hand rotating and damaged hand object detection image, so the hand gesture recognition is more robust to deal with these different condition. We got 99.925% accuracy in our model testing. The demo video is shown here (https://youtu.be/Pttl3u_ZS-s). This game design could provide user high interaction with computer and more sense of entertainment.

Algorithm flow chart:
Hand Detection -> Hand Tracking -> Gesture Recognition -> AI responds

hist: Hand color histogram
main.py: algortihm implementation
cnn_model_keras2.h5: CNN model parameters
stone.png, paper.png, scissor.png: hand gestures pictures

Run:
$ Python3 main.py
