'''
Generate curriculum given a configuration. 
Get lesson information and define the curriculum accordingly, 
which can be used by the environment.
'''

class Curriculum:
    def __init__(self, clm_idx) -> None:
        self.clm_idx = clm_idx
        self.lesson  = self.get_lesson(clm_idx)
    
    def generate_curriculum(self):
        pass
    
    def get_lesson(self):
        pass
    
    def get_config(self):
        pass