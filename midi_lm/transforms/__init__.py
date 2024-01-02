import muspy


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, music: muspy.Music) -> muspy.Music:
        for t in self.transforms:
            music = t(music)
        return music

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"\t{t}"
        format_string += "\n)"
        return format_string
