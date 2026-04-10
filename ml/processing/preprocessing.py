class PreprocessingStep:
    def process(self, data):
        return data


class Normalize(PreprocessingStep):
    def process(self, data):
        print("Normalizing data")
        return data


class Resize(PreprocessingStep):
    def process(self, data):
        print("Resizing data")
        return data