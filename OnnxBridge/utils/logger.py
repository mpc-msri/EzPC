import logging
import sys


class _AnsiColorizer(object):
    """
    A colorizer is an object that loosely wraps around a stream, allowing
    callers to write text to the stream in a particular color.

    Colorizer classes must implement C{supported()} and C{write(text, color)}.
    """

    _colors = dict(
        black=30, red=31, green=32, yellow=33, blue=34, magenta=35, cyan=36, white=37
    )

    def __init__(self, stream):
        self.stream = stream

    @classmethod
    def supported(cls, stream=sys.stdout):
        """
        A class method that returns True if the current platform supports
        coloring terminal output using this method. Returns False otherwise.
        """
        if not stream.isatty():
            return False  # auto color only on TTYs
        try:
            import curses
        except ImportError:
            return False
        else:
            try:
                try:
                    return curses.tigetnum("colors") > 2
                except curses.error:
                    curses.setupterm()
                    return curses.tigetnum("colors") > 2
            except:
                raise
                # guess false in case of error
                return False

    def write(self, text, color):
        """
        Write the given text to the stream in the given color.

        @param text: Text to be written to the stream.

        @param color: A string label for a color. e.g. 'red', 'white'.
        """
        color = self._colors[color]
        self.stream.write("\x1b[%s;3m%s\x1b[0m" % (color, text))


class ColorHandler(logging.StreamHandler):
    def __init__(self, stream=sys.stdout):
        super(ColorHandler, self).__init__(_AnsiColorizer(stream))

    def emit(self, record):
        msg_colors = {
            logging.DEBUG: "blue",
            logging.INFO: "green",
            logging.WARNING: "yellow",
            logging.ERROR: "red",
        }

        color = msg_colors.get(record.levelno, "blue")
        self.stream.write(self.format(record) + "\n", color)


class Logger:
    @classmethod
    def setup_logger(cls):
        logger = logging.getLogger("onnx-fzpc")
        console = ColorHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s <<<<- %(message)s ->>>> (%(filename)s:%(lineno)d)"
        )
        console.setFormatter(formatter)
        logger.addHandler(console)
        return logger


setup_logger = Logger.setup_logger
