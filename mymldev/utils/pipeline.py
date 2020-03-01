class Pipeline:
    def __init__(self, *parts):
        self._parts = parts

    def __or__(self, right):
        if isinstance(right, Base):
            return Pipeline(*(self._parts + (right,)))
        return NotImplemented


class Base:
    def __or__(self, right):
        if isinstance(right, Base):
            return Pipeline(self, right)
        return NotImplemented


class One(Base):
    pass


class Two(Base):
    pass


class Three(Base):
    pass


pipeline = One() | Two() | Three() | Two() | One()

pipeline._parts
