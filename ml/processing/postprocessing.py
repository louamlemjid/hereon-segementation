class PostProcessingStep:
    def process(self, output):
        return output


class Threshold(PostProcessingStep):
    def process(self, output):
        print("Applying threshold")
        return output


class ResizeBack(PostProcessingStep):
    def process(self, output):
        print("Resizing back")
        return output