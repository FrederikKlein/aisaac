import csv
import os


class ResultSaver:
    def __init__(self, context_manager):
        self.context_manager = context_manager
        self.result_path = self.context_manager.get_config('RESULT_PATH')
        self.result_file = self.context_manager.get_config('RESULT_FILE')
        self.output_file_path = os.path.join(self.result_path, self.result_file)
        self.csv_headers = ['title', 'converted', 'embedded', 'relevant', 'checkpoints', 'reasoning']
        self.chroma_path = self.context_manager.get_config('CHROMA_PATH')
        self.reset_results_bool = self.context_manager.get_config('RESET_RESULTS') == 'True'
        if self.reset_results_bool:
            self.reset_results()

    def write_csv(self, data):
        with open(self.output_file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.csv_headers)
            writer.writeheader()
            writer.writerows(data)

    def read_csv_to_dict_list(self):
        with open(self.output_file_path, 'r') as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]
        return data

    def update_csv(self, updated_data):
        existing_data = self.read_csv_to_dict_list()
        # Find the matching title and update the data
        for existing_row in existing_data:
            for updated_row in updated_data:
                if existing_row['title'] == updated_row['title']:
                    existing_row.update(updated_row)
        self.write_csv(existing_data)

    def add_data_csv(self, new_data):
        existing_data = self.read_csv_to_dict_list()
        combined_data = existing_data + new_data
        self.write_csv(combined_data)

    @staticmethod
    def __create_result_list(title):
        temp_dict = [{
            "title": title,
            "converted": True,
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

    def set_up_new_results_file(self, new_file_path, data_entries):
        # Reset the file
        self.output_file_path = new_file_path  # Update the file path for new results
        self.write_csv([])  # Create new file with empty data
        for entry in data_entries:
            title = os.path.splitext(os.path.basename(entry.metadata["source"]))[0]
            result_list = self.create_result_list(title)
            if os.path.exists(os.path.join(self.chroma_path, title)):
                result_list[0].update({"embedded": True})
            self.add_data_csv(result_list)

#%%
