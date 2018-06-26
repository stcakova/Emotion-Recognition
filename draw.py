import cv2

class DrawUtils:
    def __init__(self, frame, index):
    	self.frame = frame
    	self.index = index

    def draw_text(self, result):
	    text_color = (125, 125, 125) #Blue, Green, Red
	    padding = 10
	    diagram_start = 130
	    top_left_vertex = (diagram_start, self.index * 20 + padding)
	    emotion_percent = int(result[0][self.index] * 100)
	    bottom_right_vertex = (diagram_start + emotion_percent, (self.index + 1) * 20 + 4)
	    rectangle_filling = cv2.FILLED
	    cv2.rectangle(self.frame, top_left_vertex, bottom_right_vertex, text_color, rectangle_filling)

    def draw_diagram(self, emotion):
      	font = cv2.FONT_HERSHEY_TRIPLEX
      	diagram_color = (250, 125, 125)
      	cv2.putText(self.frame, emotion, (10, self.index * 20 + 20), font, 0.5, diagram_color, 1);