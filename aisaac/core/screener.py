from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate

from aisaac.aisaac.utils import Logger


class Screener:
    def __init__(self, context_manager):
        self.result_saver = context_manager.get_result_saver()
        self.mm = context_manager.get_model_manager()
        self.dm = context_manager.get_document_manager()
        self.similarity_searcher = context_manager.get_similarity_searcher()
        self.prompt_template = """
        [INST]
        Answer the question based only on the following context:
        
        {context}
        
        With the following checkpoints: {checkpoints}
        
        ---
        
        Answer the question based on the above context: {question}
        {format_instructions}
        [/INST]
        """
        self.question = "Which of the checkpoints are true for this document and why?"
        self.logger = Logger(__name__).get_logger()

    def do_screening(self, checkpoints):
        titles = self.dm.get_runnable_titles()
        # progress variables
        iterations = len(titles)
        counter = 0
        for title in titles:
            counter += 1
            self.logger.info(f"Processing {title} \n({counter} out of {iterations})")
            try:
                response = self.craft_screening_response_for(title, checkpoints)
                self.result_saver.save_response(response)
                self.logger.debug(f"Processed {title} successfully")
            except Exception as e:
                self.logger.error(f"Error processing {title}: {e}")

    def craft_screening_response_for(self, title, checkpoints):
        context_text = self.create_context_text(title, checkpoints)
        output_parser = self.get_output_parser()
        format_instructions = output_parser.get_format_instructions()
        prompt = self.create_prompt(context_text, checkpoints, format_instructions)
        model = self.mm.get_rag_model()
        response_text = model.predict(prompt)
        while not self.__response_correctly_formatted(response_text, output_parser):
            self.logger.info("The response was not correctly formatted. Asking again.")
            response_text = model.predict(prompt)
        data = output_parser.parse(response_text)
        return data

    def create_context_text(self, title, checkpoints):
        similarity_search_results = []
        for checkpoint in checkpoints.values():
            similarity_search_results.append(self.similarity_searcher.similarity_search(title, checkpoint))
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in similarity_search_results])
        return context_text

    def get_output_parser(self):
        response_schemas = [ResponseSchema(name="title", description="Title of the document", type="string"),
                            ResponseSchema(name="checkpoints",
                                           description="For each Checkpoint, whether it is true or false",
                                           type="string"),
                            ResponseSchema(name="reasoning", description="Reasoning for each checkpoint",
                                           type="string"), ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        return output_parser

    def __response_correctly_formatted(self, response_text, output_parser):
        try:
            data = output_parser.parse(response_text)
            return True
        except Exception as e:
            return False

    def create_prompt(self, context_text, checkpoints, format_instructions):
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        prompt = prompt_template.format(context=context_text, question=self.question,
                                        checkpoints=checkpoints, format_instructions=format_instructions)
        return prompt
