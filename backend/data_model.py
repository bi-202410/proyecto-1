from pydantic import BaseModel

class DataModel(BaseModel):
    review: str

    def columns(self):
        """
        @return list: lista de columnas correspondientes al modelo exportado 
        """
        return ["review"]


class Model(BaseModel):
    review: str
    _class: int # todo: review if this works, data columns are "review" and "class" but the model is expecting "review" and "_class"

    def columns(self):
        """
        @return list: lista de columnas correspondientes al modelo exportado 
        """
        return ["review","class"]