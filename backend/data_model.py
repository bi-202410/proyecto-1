from pydantic import BaseModel

class DataModel(BaseModel):
    review: str

    def columns(self):
        """
        @return list: lista de columnas correspondientes al modelo exportado 
        """
        return ["review"]


class TrainModel(BaseModel):
    review: str
    class_: int

    def columns(self):
        """
        @return list: lista de columnas correspondientes al modelo exportado 
        """
        return ["review","class"]