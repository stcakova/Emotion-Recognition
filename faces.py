class Face:
    def __init__(self, face):
    	self.x = face[0]
    	self.y = face[1]
        self.height = face[2]
        self.width = face[3]

    def area(self):
        return self.height * self.width

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width