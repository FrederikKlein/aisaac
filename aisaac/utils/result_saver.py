import csv
import os


class ResultSaver:
    def __init__(self, context_manager):
        self.result_path = context_manager.get_config('RESULT_PATH')
        self.result_file = context_manager.get_config('RESULT_FILE')
        self.system_manager = context_manager.get_system_manager()
        self.full_result_file_path = self.system_manager.get_full_path(
            f"{self.result_path}/{self.result_file}")
        self.system_manager.make_directory(self.result_path)
        self.csv_headers = ['title', 'converted', 'embedded', 'relevant', 'checkpoints', 'reasoning']
        self.full_chroma_path = self.system_manager.get_full_path(context_manager.get_config('CHROMA_PATH'))
        self.reset_results_bool = context_manager.get_config('RESET_RESULTS') == 'True'
        if self.reset_results_bool:
            self.reset_results()
        self.document_data_manager = context_manager.get_document_data_manager()
        if not self.system_manager.path_exists(f"{self.result_path}/{self.result_file}"):
            self.set_up_new_results_file(f"{self.result_path}/{self.result_file}",
                                         self.document_data_manager.get_all_titles())

    def write_csv(self, data):
        with open(self.full_result_file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.csv_headers)
            writer.writeheader()
            writer.writerows(data)

    def read_csv_to_dict_list(self):
        with open(self.full_result_file_path, 'r') as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]
        return data

    def read_csv_to_dict_relevant_only(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            # from the csv file, we only need the title and the relevant column
            data = {row['title']: row['relevant'] for row in reader}
        return data

    def update_csv(self, updated_data):
        existing_data = self.read_csv_to_dict_list()
        # Find the matching title and update the data
        for existing_row in existing_data:
            for updated_row in updated_data:
                if existing_row['title'] == updated_row['title']:
                    existing_row.update(updated_row)
        self.write_csv(existing_data)

    def update_result_list(self, title, key, value):
        existing_data = self.read_csv_to_dict_list()
        for existing_row in existing_data:
            if existing_row['title'] == title:
                existing_row[key] = value
        self.write_csv(existing_data)

    def add_data_csv(self, new_data):
        existing_data = self.read_csv_to_dict_list()
        combined_data = existing_data + new_data
        self.write_csv(combined_data)

    @staticmethod
    def __create_result_list(title):
        temp_dict = [{
            "title": title,
            "converted": False,
            "embedded": False,
            "relevant": None,
            "checkpoints": {},
            "reasoning": {}
        }, ]
        return temp_dict

    def create_new_result_entry(self, title):
        result_list = self.__create_result_list(title)
        self.add_data_csv(result_list)

    def get_result_list(self, title):
        result_row = {}
        # Get the result list for the title
        for row in self.read_csv_to_dict_list():
            if row['title'] == title:
                result_row = row
        temp_dict = [{
            "title": result_row['title'],
            "converted": result_row['converted'],
            "embedded": result_row['embedded'],
            "relevant": result_row['relevant'],
            "checkpoints": result_row['checkpoints'],
            "reasoning": result_row['reasoning']
        }, ]
        return temp_dict

    def reset_results(self):
        # Reset the file
        self.write_csv([])

    def set_up_new_results_file(self, new_relative_file_path, all_titles):
        converted_titles = self.document_data_manager.get_converted_titles()
        # Reset the file
        self.full_result_file_path = self.system_manager.get_full_path(
            new_relative_file_path)  # Update the file path for new results
        self.write_csv([])  # Create new file with empty data
        for title in all_titles:
            title_without_extension = self.system_manager.get_title_without_extension(title)
            result_list = self.__create_result_list(title_without_extension)
            if title_without_extension in converted_titles:
                result_list[0].update({"converted": True})
            if os.path.exists(os.path.join(self.full_chroma_path, title_without_extension)):
                result_list[0].update({"embedded": True})
            self.add_data_csv(result_list)

    def save_response(self, response, title):
        checkpoints = response['checkpoints']
        reasoning = response['reasoning']
        # in case the checkpoints are empty, we set the relevancy to None
        if len(checkpoints) == 0:
            self.update_csv([{
                "title": title,
                "relevant": None,
                "checkpoints": checkpoints,
                "reasoning": reasoning
            }])
            return

        converted_checkpoints = {key: value.lower() == 'true' if isinstance(value, str) else bool(value) for key, value
                                 in checkpoints.items()}
        # check all the values of the converted_checkpoints. If all of them are True, set relevant to True
        relevant = all(value is True for value in converted_checkpoints.values())
        # save the response to the csv file
        self.update_csv([{
            "title": title,
            "relevant": relevant,
            "checkpoints": converted_checkpoints,
            "reasoning": reasoning
        }])


# %%
