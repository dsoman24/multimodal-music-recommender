import os

class DatabaseProcessor:

    def __init__(self, data_dir, db_filename, debug_messages=False):
        self.path = os.path.join(data_dir, db_filename)
        self.df = None
        self.debug_messages = debug_messages

    def print_debug(self, message):
        if self.debug_messages:
            print(message)

    def read(self):
        """
        Reads the database file.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def process(self):
        """
        Processes the database file, initializes the df field.
        """
        raise NotImplementedError("Subclasses must implement this method.")