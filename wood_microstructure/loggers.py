import logging
import os

from rich.logging import RichHandler


class LoggerMixin:
    """Mixin class to provide a logger to the class."""
    file_fmt = logging.Formatter(
        '{asctime} - {levelname:>7s} - {message}',
        style='{'
    )
    console_fmt = None
    save_prefix = None
    logname = 'wood_microstructure'

    @staticmethod
    def ensure_dir(filename: str):
        """Ensure the directory exists"""
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)

    def __init__(self, *args, output_dir: str = None, **kwargs):
        self.init_outdir(output_dir)
        self.init_logging()
        self.handler_level = logging.DEBUG
        super().__init__(*args, **kwargs)

    def init_logging(self):
        """Initialize logging."""
        self.logger = logging.getLogger('wood' + str(self.outdir_num))
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        log_file = os.path.join(self.root_dir, f'{self.logname}.log')

        self.add_console_handler()
        self.add_file_handler(log_file)

    def add_file_handler(self, log_file_name: str):
        """Add a file logger to the logger."""
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setLevel(logging.DEBUG)
        if self.file_fmt is not None:
            file_handler.setFormatter(self.file_fmt)
        self.logger.addHandler(file_handler)

    def add_console_handler(self, level: int = logging.DEBUG):
        """Add a console logger to the logger."""
        console_handler = RichHandler(
            rich_tracebacks=True,
            tracebacks_suppress=['click', 'rich', 'rich_click', 'multiprocessing'],
            markup=True,
        )
        console_handler.setLevel(level)
        if self.console_fmt is not None:
            console_handler.setFormatter(self.console_fmt)
        self.logger.addHandler(console_handler)

    def init_outdir(self, outdir: str):
        """Initialize the output directory."""
        self.outdir = outdir or os.getenv('WOODMS_OUTDIR', '.')
        self.outdir_num = self.get_root_dir()

    def get_root_dir(self) -> int:
        """Get the root directory for saving files"""
        if self.save_prefix is None:
            raise ValueError('save_prefix must be set before calling get_root_dir()')
        dir_cnt = 0
        while os.path.exists(os.path.join(self.outdir, f'{self.save_prefix}_{dir_cnt}')):
            dir_cnt += 1
        while True:
            try:
                dir_path = os.path.join(self.outdir, f'{self.save_prefix}_{dir_cnt}')
                os.makedirs(dir_path)
            except FileExistsError:
                dir_cnt += 1
                continue
            else:
                self.root_dir = dir_path
                break
        return dir_cnt

    def set_console_level(self, level: int):
        """Set the console logging level"""
        self.handler_level = level
        for handler in self.logger.handlers:
            if isinstance(handler, RichHandler):
                handler.setLevel(level)
