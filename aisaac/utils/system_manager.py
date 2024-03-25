# make_directory(path)
# path_exists(f"{CHROMA_PATH}/{title}"):
#                 logger.info(f"Document store for {title} already exists.")


import os
import shutil


class SystemManager:
    def __init__(self, base_directory: str):
        """
        Initialize the System Manager with a base directory.

        :param base_directory: The root directory where the application's data is stored.
        """
        self.base_directory = base_directory

    def make_directory(self, relative_path: str) -> str:
        """
        Create a directory relative to the base directory if it does not already exist.

        :param relative_path: The relative path from the base directory where the directory should be created.
        :return: The full path to the created directory.
        """
        full_path = os.path.join(self.base_directory, relative_path)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def path_exists(self, relative_path: str) -> bool:
        """
        Check if a path exists relative to the base directory.

        :param relative_path: The relative path from the base directory to check.
        :return: True if the path exists, False otherwise.
        """
        full_path = os.path.join(self.base_directory, relative_path)
        return os.path.exists(full_path)

    def reset_directory(self, relative_path: str):
        """
        Delete and recreate a directory relative to the base directory.

        :param relative_path: The relative path from the base directory where the directory should be reset.
        """
        full_path = self.make_directory(relative_path)
        shutil.rmtree(full_path)
        os.makedirs(full_path)

    def save_file(self, relative_path: str, content: str, mode: str = 'wb'):
        """
        Save a file with the given content to a path relative to the base directory.

        :param relative_path: The relative path from the base directory where the file should be saved.
        :param content: The content to save into the file.
        :param mode: The mode in which the file should be opened.
        """
        full_path = os.path.join(self.base_directory, relative_path)
        with open(full_path, mode) as file:
            file.write(content)

    def get_full_path(self, relative_path: str) -> str:
        """
        Get the full path for a given relative path from the base directory.

        :param relative_path: The relative path from the base directory.
        :return: The full path.
        """
        return os.path.join(self.base_directory, relative_path)

    def delete_file(self, relative_path: str):
        """
        Delete a file relative to the base directory.

        :param relative_path: The relative path from the base directory to the file to be deleted.
        """
        full_path = self.get_full_path(relative_path)
        if os.path.isfile(full_path):
            os.remove(full_path)
