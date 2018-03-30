import cv2
import numpy as np
import pickle
import math
import sqlite3
from collections import Counter, deque
from keras.models import load_model

class Arg:
    def __init__(self, cam):
        self.resize_ratio = 0.7
        _, img = cam.read() # w = 1280, h = 720
        img = cv2.resize(img, None, fx = self.resize_ratio, fy = self.resize_ratio, interpolation = cv2.INTER_CUBIC)
        self.img_height = img.shape[0] 
        self.img_width = img.shape[1] 
        self.hand_width = int(self.img_width / 3.5 + 0.5)
        self.hand_height = self.hand_width
        self.hand_1_x = self.img_width - self.hand_width
        self.hand_1_y = self.img_height - self.hand_height
        self.hand_2_x = 0
        self.hand_2_y = self.img_height - self.hand_height
        self.debug_show = True

class Player:
    def __init__(self, w, h, x, y):
        self.hist = self.get_hand_hist()
        self.hand_width = w
        self.hand_height = h
        self.hand_x = x
        self.hand_y = y
        self.disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(w / 40), int(h / 40)))
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.start_track = False
        self.hand_x_pre = self.hand_x
        self.hand_y_pre = self.hand_y
        self.move_history = []

    def get_hand_hist(self):
        with open("hist", "rb") as f:
            hist = pickle.load(f)
        return hist

    def get_hand_result(self, img):
        hand_img = img[self.hand_y : self.hand_y + self.hand_height, self.hand_x : self.hand_x + self.hand_width]
        img_hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([img_hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
        #cv2.imshow("calcBackProject", dst)
        cv2.filter2D(dst, -1, self.disc, dst)
        blur_mask_size = self.disc.shape[0] if self.disc.shape[0] % 2 == 1 else self.disc.shape[0] + 1
        blur = cv2.GaussianBlur(dst, (blur_mask_size, blur_mask_size), 0)
        blur = cv2.medianBlur(blur, blur_mask_size)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh,thresh,thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("thresh", thresh)
        if self.start_track == False:
            contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
            if len(contours) > 0:
                contour = max(contours, key = cv2.contourArea)
                if cv2.contourArea(contour) > thresh.shape[0] * thresh.shape[1] / 6:
                    self.start_track = True

        return thresh

    def hand_track(self, img_hsv):
        if self.start_track == False:
            return
        dst = cv2.calcBackProject([img_hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
        track_window = (self.hand_x, self.hand_y, self.hand_width, self.hand_height)
        ret, track_window = cv2.meanShift(dst, track_window, self.term_crit)
        x, y, w, h = track_window
        #self.move = self.distance(self.hand_x_pre, self.hand_y_pre, x, y)
        self.hand_x_pre = self.hand_x
        self.hand_y_pre = self.hand_y
        self.hand_x = x
        self.hand_y = y

    def distance(self, x1, y1, x2, y2):  
         dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
         return dist

    def valid_play(self):
        if self.start_track == False:
            return False
        move = self.distance(self.hand_x_pre, self.hand_y_pre, self.hand_x, self.hand_y)
        self.move_history = [move] + self.move_history
        if len(self.move_history) <= 3:
            return False
        self.move_history.pop()
        avg_move = sum(self.move_history) / 3
        if avg_move <= self.hand_width / 20:
            return True
        else:
            return False

class CNN_Engine:
    def __init__(self):
        self.model = load_model('cnn_model_keras2.h5')

    def keras_process_image(self, img):
        image_x = 50
        image_y = 50
        img = cv2.resize(img, (image_x, image_y))
        img = np.array(img, dtype=np.float32)
        img = np.reshape(img, (1, image_x, image_y, 1))
        return img

    def keras_predict(self, model, image):
        processed = self.keras_process_image(image)
        pred_probab = self.model.predict(processed)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        return max(pred_probab), pred_class

    def get_pred_text_from_db(self, pred_class):
        conn = sqlite3.connect("gesture_db.db")
        cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
        cursor = conn.execute(cmd)
        for row in cursor:
            return row[0]


    def predict_play(self, hand_res):
        res = ""
        contours = cv2.findContours(hand_res.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) < hand_res.shape[0] * hand_res.shape[1] / 10:
                return res
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            save_img = hand_res[y1:y1+h1, x1:x1+w1]

            if w1 > h1:
                save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
            elif h1 > w1:
                save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
        
            pred_probab, pred_class = self.keras_predict(self.model, save_img)
            #if pred_probab * 100 > 70:
            res = self.get_pred_text_from_db(pred_class)
        return res

class AI_player:
    def __init__(self):
        self.pic_w = 150
        self.pic_h = 200
        self.paper_img = cv2.imread('paper.png')
        self.paper_img = cv2.resize(self.paper_img, (self.pic_w, self.pic_h))
        self.scissor_img = cv2.imread('scissor.png')
        self.scissor_img = cv2.resize(self.scissor_img, (self.pic_w, self.pic_h))
        self.stone_img = cv2.imread('stone.png')
        self.stone_img = cv2.resize(self.stone_img, (self.pic_w, self.pic_h))

    def ai_play(self, p1, p2):
        res = self.ai_logic(p1, p2)
        if res == "paper":
            return True, self.paper_img
        elif res == "scissor":
            return True, self.scissor_img
        elif res == "stone":
            return True, self.stone_img
        return False, []        
    
    def ai_logic(self, p1, p2):
        
        if p1 == "" or p2 == "":
            p = p1 if p2 == "" else p2
            if p == "paper":
                return "scissor"
            elif p == "scissor":
                return "stone"
            elif p == "stone":
                return "paper"
            return ""

        if p1 == "paper":
            if p2 == "paper" or p2 == "scissor":
                return "scissor"
            if p2 == "stone":
                return "paper"
        elif p1 == "scissor":
            if p2 == "stone" or p2 == "scissor":
                return "stone"
            if p2 == "paper":
                return "scissor"
        elif p1 == "stone":
            if p2 == "stone" or p2 == "paper":
                return "paper"
            if p2 == "scissor":
                return "stone"
        return ""

def main():
    
    cam = cv2.VideoCapture(0)
    arg = Arg(cam)
    recorder = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (arg.img_width, arg.img_height))
    player_1 = Player(arg.hand_width, arg.hand_height, arg.hand_1_x, arg.hand_1_y)
    player_2 = Player(arg.hand_width, arg.hand_height, arg.hand_2_x, arg.hand_2_y)
    cnn_engine = CNN_Engine()
    ai_player = AI_player()

    while cam.isOpened():
        _, img = cam.read() # w = 1280, h = 720
        img = cv2.resize(img, None, fx = arg.resize_ratio, fy = arg.resize_ratio, interpolation = cv2.INTER_CUBIC)
        img = cv2.flip(img, 1)
        p1_hand = player_1.get_hand_result(img)
        p1_play = cnn_engine.predict_play(p1_hand)
        p2_hand = player_2.get_hand_result(img)
        p2_play = cnn_engine.predict_play(p2_hand)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        player_1.hand_track(img_hsv)
        player_2.hand_track(img_hsv)

        valid, ai_res = ai_player.ai_play(p1_play, p2_play)
        if valid and (player_1.valid_play() and player_2.valid_play() or (not player_1.start_track) and player_2.valid_play() or (not player_2.start_track) and player_1.valid_play()):
            middle = int((img.shape[1] - ai_player.pic_w) / 2 + 0.5)
            img[0 :ai_player.pic_h, middle : middle + ai_player.pic_w] = ai_res
        if arg.debug_show == True:
            cv2.putText(img, "Player2: " + p1_play, (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 100, 255))
            cv2.putText(img, "Player1: " + p2_play, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 100, 255))
            cv2.rectangle(img, (player_1.hand_x, player_1.hand_y), (player_1.hand_x + player_1.hand_width, player_1.hand_y + player_1.hand_height), (0, 0, 255), 2)
            cv2.rectangle(img, (player_2.hand_x, player_2.hand_y), (player_2.hand_x + player_2.hand_width, player_2.hand_y + player_2.hand_height), (0, 0, 255), 2)
        cv2.imshow("p2_hand", p1_hand)
        cv2.imshow("p1_hand", p2_hand)
        if player_1.start_track == True:
            cv2.putText(img,'Player2', (player_1.hand_x, player_1.hand_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 100, 0), 2, cv2.LINE_AA)
        if player_2.start_track == True:
            cv2.putText(img,'Player1', (player_2.hand_x, player_2.hand_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (177, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("paper scissor stone", img)
        recorder.write(img)
        keyboard_input = cv2.waitKey(50)
        if keyboard_input == ord('q'):
            break
        if p1_play == "":
            player_1.hand_x = arg.hand_1_x
            player_1.hand_y = arg.hand_1_y
            player_1.start_track = False

        if p2_play == "":
            player_2.hand_x = arg.hand_2_x
            player_2.hand_y = arg.hand_2_y
            player_2.start_track = False
                
        if keyboard_input == ord('r'):
            recorder = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (arg.img_width, arg.img_height))
            player_1.hand_x = arg.hand_1_x
            player_1.hand_y = arg.hand_1_y
            player_1.start_track = False
            player_2.hand_x = arg.hand_2_x
            player_2.hand_y = arg.hand_2_y
            player_2.start_track = False

    cam.release()
    recorder.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()