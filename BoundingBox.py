

class BoundingBoxVolume(PyScriptObject):

    def __init__(self):
        self.data.visible = True

    def update(self):
        pass

    def compute(self):
        if self.data.source() is None:
            return
        
        data = self.data.source()
        min_bounds, max_bounds = data.bounding_box
        x_extend = max_bounds[0] - min_bounds[0]
        y_extend = max_bounds[1] - min_bounds[1]
        z_extend = max_bounds[2] - min_bounds[2]

        volume = x_extend * y_extend * z_extend

        print(f'The volume of {data.name} is {volume}. ')
    



