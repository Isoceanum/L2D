import Box2D
from constants import L2D_CATEGORY_WALL

class RayCastCallback(Box2D.b2RayCastCallback):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length
        self.hit = False
        self.distance = max_length  # Default: no hit → return full length

    def ReportFixture(self, fixture, point, normal, fraction):        
        if fixture.filterData.categoryBits != L2D_CATEGORY_WALL:
            return -1  # skip this fixture
            
        self.hit = True
        self.distance = fraction * self.max_length
        return fraction

