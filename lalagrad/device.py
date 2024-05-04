
from typing import Final

class Device:
    def __init__(self, name:str): self.name = name
    def __repr__(self): return f"<device: {self.name}>"
    
class devices:
    CPU: Final[Device] = Device('CPU')   
    GPU: Final[Device] = Device('GPU')