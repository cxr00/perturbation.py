"""
https://github.com/cxr00/perturbation.py

by Conrad
"""

# If you can't import it, that's fine
try:
    from PIL import Image
except:
    pass

import pygame
import random
import math

# Range for non-whitespace characters
min_char = 33
max_char = 126

# Event types
SCROLL = 0
BROADCAST = 1

# Modes for ScrollEvents
SCROLLEVENT_MODES = ["+V", "+H", "+VH", "-V", "-H", "-VH"]


def get_color():
    """
    The default color for PerturbationTexts
    """
    return 0, 255, 255


def random_ascii_char():
    """
    Get a random non-whitespace character
    """
    return chr(random.randint(min_char, max_char))


def _pad_string(s_i, width, left_padding=0):
    pad_length = width - len(s_i) - left_padding
    if pad_length < 0:
        output = [" " for _ in range(left_padding)] + [s_i[:pad_length]]
    else:
        output = [" " for _ in range(left_padding)] + [*s_i] + [" " for _ in range(pad_length)]
    output = "".join(output)
    return output


def _get_ascii_representation(img_path):
    """
    Convert an image at the given path to its ASCII representation for a PerturbationMatrix

    This function also provides the width and height of the encoded image
    """
    image = Image.open(img_path)

    width, height = image.size
    pixels = list(image.getdata())
    output = [[] for _ in range(height)]

    for i in range(height):
        for j in range(width):
            pixel = pixels[i*width + j]
            rgb_sum = sum(pixel[:-1]) / 3  # Average of pixel values
            rgb_sum = int(rgb_sum // 2.75)  # Scale down to between 0 and 93
            output[i].append(chr(min_char + rgb_sum))

    return output, width, height


def _get_string_list(strings, width, height, left_padding=0):
    """
    Turn a string or list of strings into a character set for a PerturbationMatrix
    """
    if strings is None:
        strings = [""]

    if isinstance(strings, str):
        strings = strings.split("\n")

    for i in range(len(strings)):
        strings[i] = _pad_string(strings[i], width, left_padding)

    l = height - len(strings)
    for i in range(l):
        strings = [_pad_string("", width, left_padding)] + strings

    return strings, width, height


class ScrollEvent:
    """
    ScrollEvents indicate a type of structured perturbation of a PerturbationMatrix

    EVENT MODES
    +V - scroll vertically downwards
    -V - scroll vertically upwards
    +H - scroll horizontally to the right
    -H - scroll horizontally to the left
    +VH - scroll diagonally down to the right
    -VH - scroll diagonally up to the left
    """

    @staticmethod
    def generate(master):
        """
        Create a ScrollEvent for the given PerturbationMatrix
        """

        height = master.height
        width = master.width

        text = "".join([random_ascii_char() for _ in range(random.randint(2, 10))])
        scroll_type = random.choice(SCROLLEVENT_MODES)

        if "+" in scroll_type:
            row = random.randint(0, height - 1) if "H" in scroll_type else -1
            column = random.randint(0, width - 1) if "V" in scroll_type else -1
            if "VH" in scroll_type:
                if random.randint(0, 1):
                    row = -1
                else:
                    column = -1
        else:
            row = height - random.randint(0, height) if "H" in scroll_type else height
            column = width - random.randint(0, width) if "V" in scroll_type else width
            if "VH" in scroll_type:
                if random.randint(0, 1):
                    row = height
                else:
                    column = width

        return ScrollEvent(master, text, scroll_type, row, column)

    def __init__(self, master, text, scroll_type, row=-1, column=-1, after=None):
        self.master = master
        self.text = text
        self.type = SCROLL
        self.mode = scroll_type
        self.row = row
        self.column = column
        self.after_func = after

    def __str__(self):
        return f"ScrollEvent: {self.mode}, {self.text}, {self.row}, {self.column}"

    def after(self):
        if self.after_func is not None:
            self.after_func()

    def update(self):
        if "+" in self.mode:
            if "H" in self.mode:
                self.column += 1
            if "V" in self.mode:
                self.row += 1
        elif "-" in self.mode:
            if "H" in self.mode:
                self.column -= 1
            if "V" in self.mode:
                self.row -= 1
        else:
            raise ValueError(f"Invalid ScrollEvent mode {self.mode}")

    def is_complete(self):
        return not (self.master.height > self.row >= 0) or not (self.master.width > self.column >= 0)


class BroadcastEvent:
    """
    A BroadcastEvent indicates the placement of static text for a sustained period of time
    """

    @staticmethod
    def generate(master):
        text = ["".join([random_ascii_char() for _ in range(random.randint(3, 10))]) for __ in range(random.randint(1, 3))]
        row = random.randint(0, master.height - 1)
        column = random.randint(0, master.width - 1)
        time = 100
        return BroadcastEvent(text, column, row, time)

    def __init__(self, text, column, row, time, after=None):
        if isinstance(text, str):
            text = text.split("\n")
        self.text = text

        # Wrap the text for readability
        l = max(len(k) for k in self.text)
        self.text = ["*" * (l + 2)] + ["*" + " " * l + "*"] + ["*" + k + " " * (l - len(k)) + "*" for k in self.text] + ["*" + " " * l + "*"] + ["*" * (l + 2)]

        self.row = row
        self.column = column
        self.type = BROADCAST
        self.time = time
        self.after_func = after

    def __str__(self):
        return f"BroadcastEvent: {self.type} {self.column} {self.row} {self.time}"

    def after(self):
        if self.after_func is not None:
            self.after_func()

    def update(self):
        self.time -= 1
        if self.time < 0:
            self.time = 0

    def is_complete(self):
        return self.time == 0


class PerturbationText:
    """
    A PerturbationText draws a stream of random characters that ultimately resolves to a specific character.

    There is also a small chance that the character will be further randomized after it has already resolved
    """

    def __init__(self, master, font, x_offset, y_offset, character, magnitude, text_color):
        self.master = master
        self.font = font
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.character = character
        self.perturbation_length = 0 if magnitude == 0 else random.randint(1, magnitude)
        self.current_char = random_ascii_char()
        self.text_color = text_color

    def update(self):
        if self.perturbation_length:
            self.perturbation_length -= 1
            self.current_char = random_ascii_char()
        else:
            self.current_char = self.character

    def set(self, char):
        """
        Temporarily set the character
        """
        self.current_char = char

    def change(self, char):
        """
        Change the character's true form
        """
        self.character = char

    def perturb(self, magnitude, chance):
        """
        Cause the text to randomize temporarily
        """
        if random.randint(0, chance) == chance:
            self.perturbation_length += random.randint(1, magnitude)

    def draw(self, x, y, i, j):
        text = self.font.render(self.current_char, True, self.text_color)
        text_rect = text.get_rect()
        text_rect.center = x + i * self.x_offset, y + j * self.y_offset
        self.master.blit(text, text_rect)


class PerturbationMatrix:
    """
    PerturbationMatrix manages a collection of PerturbationTexts which form an ASCII image or Qoid representation
    """

    def __init__(self, master, font, x_offset, y_offset, img_path=None, string_value=None, dim=None, text_color=None):
        self.master = master
        self.font = font
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.text_color = text_color if text_color is not None else get_color()

        if img_path is None and (dim is None or not (isinstance(dim, tuple) and len(dim) == 2)):
            raise ValueError(f"Please specify a valid dimension for your PerturbationMatrix")
        elif img_path is not None:
            self.text, self.width, self.height = _get_ascii_representation(img_path)
        else:
            self.text, self.width, self.height = _get_string_list(string_value, dim[0], dim[1])

        self.perturbationtexts = None
        self.events = []
        self.linequeue = []
        self.initialize()

    def __getitem__(self, item):
        return self.perturbationtexts[item]

    def __iadd__(self, other):
        if other is None:
            other = [None]
        if isinstance(other, str):
            other = other.split("\n")

        if isinstance(other, (list, tuple)):
            for line in other:
                if line is not None and not isinstance(line, str):
                    raise TypeError(f"Invalid line type {type(line)}")
                if line is not None:
                    line = line.split("\n")
                    self.linequeue += line
                else:
                    self.linequeue += [""]
        else:
            raise TypeError(f"Can only add None, str, list, or tuple to PerturbationMatrix, not {type(other)}")
        return self

    def initialize(self):
        self.perturbationtexts = [[] for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                self[i].append(PerturbationText(self.master, self.font, self.x_offset, self.y_offset, self.text[i][j], 0, self.text_color))

    def clear_events(self, type=-1):
        if type != -1:
            n = len(self.events) - 1
            while n >= 0:
                if self.events[n].type == type:
                    self.events.pop(n)
                n -= 1
        else:
            self.events = []

    def clear_perturbations(self):
        for i in range(self.height):
            for j in range(self.width):
                self[i][j].perturbation_length = 0

    def empty(self):
        self.text = _get_string_list([""], self.width, self.height)[0]
        for i in range(self.height):
            for j in range(self.width):
                self[i][j].change(self.text[i][j])

    def update(self):
        # Update individual PerturbationTexts
        for i in range(self.height):
            for j in range(self.width):
                self[i][j].update()

        # Process events
        for event in self.events:
            event.update()
            if not event.is_complete():
                if event.type == SCROLL:
                    for i in range(len(event.text)):
                        if "-" in event.mode:
                            i = -i
                        if "VH" in event.mode:
                            if 0 <= (event.row + i) < self.height and 0 <= (event.column + i) < self.width:
                                self[event.row + i][event.column + i].set(event.text[abs(i)])
                        elif "H" in event.mode:
                            if 0 <= (event.column + i) < self.width:
                                self[event.row][event.column + i].set(event.text[abs(i)])
                        elif "V" in event.mode:
                            if 0 <= (event.row + i) < self.height:
                                self[event.row + i][event.column].set(event.text[abs(i)])
                elif event.type == BROADCAST:
                    for i in range(len(event.text)):
                        for j in range(len(event.text[i])):
                            if 0 <= (event.row + i) < self.height and 0 <= (event.column + j) < self.width:
                                self[event.row + i][event.column + j].set(event.text[i][j])

        # Update line queue
        if self.linequeue:
            self.feed_line(self.linequeue.pop(0))

        n = len(self.events) - 1
        while n >= 0:
            if self.events[n].is_complete():
                self.events.pop(n).after()
            n -= 1

    def draw(self, x, y):
        for i in range(self.width):
            for j in range(self.height):
                # Note the transposed access of j, i instead of i, j. This tripped me up for a while.
                self[j][i].draw(x + self.x_offset, y + self.y_offset, i, j)

    def get_size(self):
        return (self.width + 1) * self.x_offset, (self.height + 1) * self.y_offset

    @staticmethod
    def calculate_size(dim, x_offset, y_offset):
        return (dim[0] + 1) * x_offset, (dim[1] + 1) * y_offset

    def translate(self):
        """
        Randomly modularly translate the PerturbationMatrix
        """
        up, down, left, right = (random.randint(0, 1) for _ in range(4))

        if not (up and down):
            if up:
                # Move every row up one, and move the top row to the bottom
                for i in range(self.height):
                    new_last = self[i][0]
                    for j in range(self.width - 2):
                        self[i][j], self[i][(j + 1) % self.width] = self[i][(j + 1) % self.width], self[i][(j + 2) % self.width]
                    self[i][self.width - 1] = new_last
            elif down:
                # Move every row down one, and move the bottom row to the top
                for i in range(self.height):
                    new_first = self[i][self.width - 1]
                    for j in range(self.width - 1, 0, -1):
                        self[i][j], self[i][(j - 1) % self.width] = self[i][(j - 1) % self.width], self[i][(j - 2) % self.width]
                    self[i][0] = new_first

        if not (left and right):
            if left:
                # Move every column left one, and move the leftmost column to the right
                self.perturbationtexts = self[1:] + [self[0]]
            elif right:
                # Move every column right one, and move the rightmost column to the left
                self.perturbationtexts = [self[-1]] + self[:-1]

    def perturb(self, magnitude, chance):
        for i in range(self.height):
            for j in range(self.width):
                self[i][j].perturb(magnitude, chance)

    def add_event(self, e):
        self.events.append(e)

    def add_scrollevent(self):
        se = ScrollEvent.generate(self)
        self.events.append(se)

    def add_broadcastevent(self):
        be = BroadcastEvent.generate(self)
        self.events.append(be)

    def set_text(self, text):
        self.text = text
        for i in range(self.height):
            for j in range(self.width):
                self[i][j].change(self.text[i][j])

    def feed_line(self, string_value):
        """
        Feed a line to the bottom of the PerturbationMatrix
        """
        self.text += [_pad_string(string_value, self.width)]
        self.set_text(self.text[1:])

    def is_normal(self):
        """
        Return whether the PerturbationMatrix is in a neutral state, ie has no events or current perturbations
        """
        if self.events:
            return False
        for i in range(self.height):
            for j in range(self.width):
                if self[i][j].perturbation_length:
                    return False
        return True

    def rotate(self, clockwise=True):
        """
        Rotate the PerturbationMatrix if its width and height are equal

        Notice the way the algorithm reverses when comparing clockwise vs anticlockwise
        """
        if self.width != self.height:
            raise ValueError(f"Cannot rotate PerturbationMatrix with unequal dimensions ({self.width}, {self.height})")
        if clockwise:
            for i in range(self.height // 2):
                for j in range(i, self.width - i - 1):
                    tmp = self[self.width - j - 1][i]
                    self[self.width - j - 1][i] = self[self.width - i - 1][self.height - j - 1]
                    self[self.width - i - 1][self.height - j - 1] = self[j][self.height - i - 1]
                    self[j][self.height - i - 1] = self[i][j]
                    self[i][j] = tmp
        else:
            for i in range(self.height // 2):
                for j in range(i, self.width - i - 1):
                    tmp = self[i][j]
                    self[i][j] = self[j][self.height - i - 1]
                    self[j][self.height - i - 1] = self[self.width - i - 1][self.height - j - 1]
                    self[self.width - i - 1][self.height - j - 1] = self[self.width - j - 1][i]
                    self[self.width - j - 1][i] = tmp

    def random_pulse(self):
        r = random.randint(0, 2)
        if r == 0:
            self.box_pulse()
        elif r == 1:
            self.diamond_pulse()
        elif r == 2:
            self.circle_pulse()

    def box_pulse(self):
        x_r = random.randint(0, self.width)
        y_r = random.randint(0, self.height)

        pulse_range = random.randint(1, 4)
        delay = random.randint(1, 5)

        for i in range(-pulse_range, pulse_range + 1):
            for j in range(-pulse_range, pulse_range + 1):
                if self.height > (y_r + i) >= 0 and self.width > (x_r + j) >= 0:
                    self[y_r + i][x_r + j].perturb(max([abs(i), abs(j)]) + delay, 0)

    def diamond_pulse(self):
        x_r = random.randint(0, self.width)
        y_r = random.randint(0, self.height)

        pulse_range = random.randint(1, 5)
        delay = random.randint(1, 5)

        for i in range(-pulse_range, pulse_range + 1):
            for j in range(-pulse_range, pulse_range + 1):
                if self.height > (y_r + i) >= 0 and self.width > (x_r + j) >= 0:
                    if abs(i) + abs(j) <= pulse_range:
                        self[y_r + i][x_r + j].perturb(abs(i) + abs(j) + delay, 0)

    def circle_pulse(self):
        x_r = random.randint(0, self.width)
        y_r = random.randint(0, self.height)

        radius = random.randint(1, 5)
        delay = random.randint(1, 5)

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if self.height > (y_r + i) >= 0 and self.width > (x_r + j) >= 0:
                    if int(math.sqrt(i**2 + j**2)) < radius:
                        self[y_r + i][x_r + j].perturb(int(math.sqrt(i**2 + j**2)) + delay, 0)


def perturbation_matrix_test():
    pygame.init()
    dim = (50, 50)
    x_offset = 10
    y_offset = 10
    font = pygame.font.Font(None, 18)
    screen = pygame.display.set_mode(PerturbationMatrix.calculate_size(dim, x_offset, y_offset))
    matrix = PerturbationMatrix(screen, font, x_offset, y_offset, img_path="img/test.png")

    perturbation_rate = 10
    translation_rate = 10
    scroll_rate = 3
    pulse_rate = 5
    broadcast_rate = 10

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exit(0)
            if event.type == pygame.KEYDOWN:
                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_r]:
                    matrix.initialize()
                if pressed[pygame.K_DOWN]:
                    if perturbation_rate < 30:
                        perturbation_rate += 1
                        print("perturbation:", perturbation_rate)
                if pressed[pygame.K_UP]:
                    if perturbation_rate > 0:
                        perturbation_rate -= 1
                        print("perturbation:", perturbation_rate)
                if pressed[pygame.K_RIGHT]:
                    if translation_rate > 0:
                        translation_rate -= 1
                        print("translation:", translation_rate)
                if pressed[pygame.K_LEFT]:
                    if translation_rate < 30:
                        translation_rate += 1
                        print("translation:", translation_rate)
                if pressed[pygame.K_v]:
                    if scroll_rate > 0:
                        scroll_rate -= 1
                        print("scroll:", scroll_rate)
                if pressed[pygame.K_c]:
                    if scroll_rate < 5:
                        scroll_rate += 1
                        print("scroll:", scroll_rate)

        screen.fill((0, 0, 0))
        matrix.update()

        # Add perturbations
        if random.randint(0, translation_rate) == translation_rate:
            matrix.translate()
        if random.randint(0, perturbation_rate) == perturbation_rate:
            matrix.perturb(10, 25)
        if random.randint(0, scroll_rate) == scroll_rate:
            matrix.add_scrollevent()
        if random.randint(0, pulse_rate) == pulse_rate:
            matrix.random_pulse()
        if random.randint(0, broadcast_rate) == broadcast_rate:
            matrix.add_broadcastevent()

        # Draw matrix
        matrix.draw(0, 0)

        pygame.display.update()
        pygame.time.wait(50)  # 60 FPS


if __name__ == "__main__":
    perturbation_matrix_test()
