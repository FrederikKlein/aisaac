from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate


class CriteriaOptimizer:
    def __init__(self, context_manager):
        self.logger = context_manager.get_logger(__name__)
        self.result_saver = context_manager.get_result_saver()
        self.mm = context_manager.get_model_manager()
        self.dm = context_manager.get_document_manager()
        self.similarity_searcher = context_manager.get_similarity_searcher()
        self.feature_importance_threshold = float(context_manager.get_config('FEATURE_IMPORTANCE_THRESHOLD'))
        self.max_documents = int(context_manager.get_config('MAX_FEATURE_IMPROVEMENT_DOCUMENTS'))
        self.importance_greater_than_threshold = context_manager.get_config('IMPORTANCE_GREATER_THAN_THRESHOLD')
        self.number_expert_choices = int(context_manager.get_config('NUMBER_EXPERT_CHOICES'))

    # This function is a simple way to improve the feature importance of a model. The importance_threshold is strictly
    # greater than or strictly less than, depending on the importance_greater_than_threshold parameter
    def automated_feature_improvement(self, checkpoint_dictionary, checkpoint_importances):
        checkpoint_keys = list(checkpoint_dictionary.keys())
        for i, importance in enumerate(checkpoint_importances):
            if (importance > self.feature_importance_threshold) == self.importance_greater_than_threshold:
                current_checkpoint = checkpoint_dictionary[checkpoint_keys[i]]
                new_checkpoint = self.__generate_new_checkpoints(current_checkpoint, k=1)[0]
                self.logger.debug(f"Old checkpoint: {current_checkpoint}")
                self.logger.debug(f"New checkpoint: {new_checkpoint}")
                checkpoint_dictionary[checkpoint_keys[i]] = new_checkpoint
        return checkpoint_dictionary

    # This function generates more than one possible new checkpoint and asks the expert to choose one
    # It also allows for annotations from the expert, which might come in handy
    def expert_feature_improvement(self, checkpoint_dictionary, checkpoint_importances, annotations=None):
        checkpoint_keys = list(checkpoint_dictionary.keys())
        for i, importance in enumerate(checkpoint_importances):
            if (importance > self.feature_importance_threshold) == self.importance_greater_than_threshold:
                new_checkpoints = self.__expert_feature_improvement(checkpoint_dictionary[checkpoint_keys[i]],
                                                                    notes=annotations)
                checkpoint_dictionary[checkpoint_keys[i]] = self.__ask_for_expert_choice(new_checkpoints)
        return checkpoint_dictionary

    # This function generates more than one possible new checkpoint based on the context of the current checkpoint
    # An LLM will decide which of the new checkpoints is the best one or averages one
    def context_aware_feature_improvement(self, checkpoint_dictionary, checkpoint_importances):
        checkpoint_keys = list(checkpoint_dictionary.keys())
        for i, importance in enumerate(checkpoint_importances):
            if (importance > self.feature_importance_threshold) == self.importance_greater_than_threshold:
                current_checkpoint = checkpoint_dictionary[checkpoint_keys[i]]
                potential_checkpoints = []
                titles_with_limit = self.__get_runnable_title_with_limit()
                for title in titles_with_limit:
                    context_text = self.similarity_searcher.similarity_search(title, current_checkpoint)
                    checkpoint_candidate = self.__generate_improved_checkpoints(current_checkpoint, context_text)
                    self.logger.debug(f"Potential new checkpoint: {checkpoint_candidate}")
                    potential_checkpoints.append(checkpoint_candidate)
                checkpoint_dictionary[checkpoint_keys[i]] = self.__average_checkpoints(potential_checkpoints)
        return checkpoint_dictionary

    # The most advanced feature improvement method
    # We ask an LLM to return an improved checkpoint based on the context of the current checkpoint, the expert's notes and the most important text chunks for every document that the expert has labeled. Based on the returned checkpoints, the LLM creates a new one, which is then presented to the expert for approval. Wo do that for every checkpoint that has an importance greater than the threshold
    def advanced_feature_improvement(self, annotations, checkpoint_dictionary, checkpoint_importances):
        checkpoint_keys = list(checkpoint_dictionary.keys())

        # for every important checkpoint do:
        # for every document (maxed) get the context text and generate improved checkpoints
        # get expert choice or get LLM choice
        # return the improved checkpoints

        checkpoint_keys = list(checkpoint_dictionary.keys())
        for i, importance in enumerate(checkpoint_importances):
            if (importance > self.feature_importance_threshold) == self.importance_greater_than_threshold:
                current_checkpoint = checkpoint_dictionary[checkpoint_keys[i]]
                potential_checkpoints = []
                titles_with_limit = self.__get_runnable_title_with_limit()
                for title in titles_with_limit:
                    context_text = self.similarity_searcher.similarity_search(title, current_checkpoint)
                    checkpoint_candidate = self.__generate_improved_checkpoints(current_checkpoint=current_checkpoint,
                                                                                context_text=context_text,
                                                                                annotations=annotations)
                    self.logger.debug(f"Potential new checkpoint: {checkpoint_candidate}")
                    potential_checkpoints.append(checkpoint_candidate)
                expert_presentable_checkpoints = []
                # divide the potential checkpoints into k groups where k is the number of choices the expert wants to have
                potential_checkpoint_groups = [potential_checkpoints[i:i + self.number_expert_choices] for i in
                                               range(0, len(potential_checkpoints), self.number_expert_choices)]
                # loop k times
                for i in range(self.number_expert_choices):
                    expert_presentable_checkpoints.append(self.__average_checkpoints(potential_checkpoint_groups[i]))
                checkpoint_dictionary[checkpoint_keys[i]] = self.__ask_for_expert_choice(expert_presentable_checkpoints)
        return checkpoint_dictionary

    def __ask_for_expert_choice(self, new_checkpoints):
        while True:
            try:
                choice_string = "[" + "/".join(map(str, range(1, len(new_checkpoints) + 1))) + "]"
                options_string = "\n".join(f"[{i + 1}] {checkpoint}" for i, checkpoint in enumerate(new_checkpoints))
                choice = int(input(
                    f"Please name which of the propositions you want to keep. {choice_string} \n Here are the options: \n {options_string}"))
                if 1 <= choice <= len(new_checkpoints):
                    break
                else:
                    print(f"Please enter a number from 1 to {len(new_checkpoints)}.")
            except ValueError:
                print("Please enter a valid number.")

        print(f"You chose checkpoint {choice}: {new_checkpoints[choice - 1]}")
        return new_checkpoints[choice - 1]

    def check_propositions(self, current_checkpoint, new_checkpoints):
        while True:
            try:
                checkpoints_string = "\n".join(
                    f"[{i + 1}] {checkpoint}" for i, checkpoint in enumerate(new_checkpoints))
                choice = input(
                    f"Here are some propositions for improving the {current_checkpoint} checkpoint. \nIs there "
                    f"anything that you would like to add? [y/n] \n {checkpoints_string}")
                if choice == "y":
                    user_annotations = input("Please enter your annotations freely")
                    # create new propositions with the user annotations
                    return self.__expert_feature_improvement(current_checkpoint, notes=user_annotations)

                if choice == "n":
                    return new_checkpoints
                else:
                    print(
                        f"Please only state whether you want to add something or not by writing 'y' for yes or 'n' for no.")
            except ValueError:
                print(
                    f"Please only state whether you want to add something or not by writing 'y' for yes or 'n' for no.")

    def __expert_feature_improvement(self, current_checkpoint, notes=None):
        new_checkpoints = self.__generate_new_checkpoints(current_checkpoint, notes=notes)
        self.logger.debug(f"generated checkpoints: {new_checkpoints}")
        return self.check_propositions(current_checkpoint, new_checkpoints)

    def __generate_new_checkpoints(self, current_checkpoint, k=None, notes=None):
        if k is None:
            k = self.number_expert_choices
        new_checkpoints = []
        for i in range(k):
            new_checkpoints.append(
                self.__generate_improved_checkpoint_without_context(current_checkpoint=current_checkpoint,
                                                                    annotations=notes))
        return new_checkpoints

    def __generate_improved_checkpoints(self, current_checkpoint, context_text, annotations):
        prompt_text_template = """
        [INST]
        Answer the question based only on the following context:
        
        {context}
        
        With the following checkpoint: {checkpoint}
        
        And the following expert annotations: {annotations}
        
        ---
        
        Answer the question based on the above context: {question}
        {format_instructions}
        [/INST]
        """
        question = "What is a better checkpoint?"

        output_parser = self.__get_output_parser()
        format_instructions = output_parser.get_format_instructions()
        prompt_template = ChatPromptTemplate.from_template(prompt_text_template)
        prompt = prompt_template.format(context=context_text, question=question, checkpoint=current_checkpoint,
                                        annotations=annotations, format_instructions=format_instructions)
        self.logger.debug(prompt)
        model = self.mm.get_rag_model()
        response_text = model.predict(prompt)
        while not self.__response_correctly_formatted(response_text, output_parser):
            self.logger.info("The response was not correctly formatted. Asking again.")
            response_text = model.predict(prompt)
        data = output_parser.parse(response_text)
        first_key = next(iter(data))
        self.logger.debug(f"Improved checkpoint: {data[first_key]}")
        data = data[first_key]
        self.logger.debug(data)

        return data

    def __generate_improved_checkpoint_without_context(self, current_checkpoint, annotations):
        prompt_text_template = """
        [INST]
        A checkpoint is a string describing the criteria for including or excluding a piece of data in a dataset.
        Here is a suboptimal checkpoint:
        
        {checkpoint}
        
        Please provide a better one considering the following expert annotations if there are any: {annotations}
        
        ---
    
        {format_instructions}
        [/INST]
        """
        output_parser = self.__get_output_parser()
        format_instructions = output_parser.get_format_instructions()

        prompt_template = ChatPromptTemplate.from_template(prompt_text_template)
        prompt = prompt_template.format(checkpoint=current_checkpoint, annotations=annotations,
                                        format_instructions=format_instructions)
        self.logger.debug(prompt)
        model = self.mm.get_rag_model()
        response_text = model.predict(prompt)
        while not self.__response_correctly_formatted(response_text, output_parser):
            self.logger.info("The response was not correctly formatted. Asking again.")
            response_text = model.predict(prompt)
        data = output_parser.parse(response_text)
        first_key = next(iter(data))
        self.logger.debug(f"Improved checkpoint: {data[first_key]}")
        data = data[first_key]
        self.logger.debug(data)

        return data

    def __average_checkpoints(self, checkpoints):
        prompt_text_template = """
        [INST]
        A checkpoint is a string describing the criteria for including or excluding a piece of data in a dataset.
        Here is a list of checkpoints that might be a better for the problem at hand. 
        Create the best checkpoint based on the following checkpoints: 
        {checkpoints}
        
        {format_instructions}
        [/INST]
        """

        output_parser = self.__get_output_parser()
        format_instructions = output_parser.get_format_instructions()

        prompt_template = ChatPromptTemplate.from_template(prompt_text_template)
        prompt = prompt_template.format(checkpoints=checkpoints, format_instructions=format_instructions)
        self.logger.debug(prompt)
        model = self.mm.get_rag_model()
        response_text = model.predict(prompt)
        while not self.__response_correctly_formatted(response_text, output_parser):
            self.logger.info("The response was not correctly formatted. Asking again.")
            response_text = model.predict(prompt)
        data = output_parser.parse(response_text)
        first_key = next(iter(data))
        self.logger.debug(f"Averaged checkpoint: {data[first_key]}")
        data = data[first_key]
        self.logger.debug(data)

        return data

    def __response_correctly_formatted(self, response, output_parser):
        try:
            data = output_parser.parse(response)
            return True
        except:
            return False

    def __get_output_parser(self):
        response_schemas = [
            ResponseSchema(name="improved checkpoint", description="The description of the checkpoint", type="string"),
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)

    # gets the first MAX_DOCUMENTS amount of runnable titles
    def __get_runnable_title_with_limit(self):
        return self.dm.get_runnable_titles()[:self.max_documents]
