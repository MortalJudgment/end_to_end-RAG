from langchain_core.prompts import ChatPromptTemplate
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains import create_extraction_chain_pydantic
from dotenv import load_dotenv

load_dotenv()

class AgenticChunker:
    def __init__(self, api_key=None):
        self.chunks = {}
        self.id_truncate_limit = 5

        # Whether or not to update/refine summaries and titles as you get new information
        self.generate_new_metadata_ind = True
        self.print_logging = True

        if api_key is None:
            # api_key = os.getenv("GROQ_API_KEY")
            api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("API key is not provided and not found in environment variables")

        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.3)
        # self.llm = ChatGroq(model="llama3-70b-8192", api_key=api_key, temperature=0.3)

    def add_contexts(self, contexts):
        for context in contexts:
            self.add_context(context)
    
    def add_context(self, context):
        if self.print_logging:
            print(f"\nAdding: '{context}'")

        # If it's your first chunk, just make a new chunk and don't check for others
        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(context)
            return

        chunk_id = self._find_relevant_chunk(context)

        # If a chunk was found then add the context to it
        if chunk_id:
            if self.print_logging:
                print(f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_context_to_chunk(chunk_id, context)
            return
        else:
            if self.print_logging:
                print("No chunks found")
            # If a chunk wasn't found, then create a new one
            self._create_new_chunk(context)
        

    def add_context_to_chunk(self, chunk_id, context):
        # Add then
        self.chunks[chunk_id]['contexts'].append(context)

        # Then grab a new summary
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the summary or else they could get stale
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an AI tasked with maintaining and summarizing groups of related sentences (chunks). Your role is to generate concise, one-sentence summaries that accurately represent the content of each chunk.

                    Guidelines:
                    1. Create a brief, informative summary that captures the essence of the chunk.
                    2. Generalize concepts when possible (e.g., "apples" to "food", "July" to "date and time").
                    3. Include the main topic and any relevant subtopics.
                    4. If applicable, mention what kind of information should be added to the chunk.
                    5. Ensure the summary is clear and easily understandable.

                    You will receive:
                    1. A list of propositions (sentences) in the chunk.
                    2. The current summary of the chunk.

                    Your task is to create an updated summary based on this information.

                    Example:
                    Input: 
                    Proposition: Greg likes to eat pizza
                    Current summary: This chunk contains information about Greg's preferences.
                    
                    Output: This chunk discusses Greg's food preferences, particularly his enjoyment of various dishes.

                    Respond only with the new summary, without any additional text or explanations.
                    """,
                ),
                ("user", "Chunk's contexts:\n{context}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "context": "\n".join(chunk['contexts']),
            "current_summary": chunk['summary']
        }).content

        return new_chunk_summary
    
    def _update_chunk_title(self, chunk):
        """
        Updates the chunk title when new context is added to ensure it remains relevant and accurate.
        """
        PROMPT = ChatPromptTemplate.from_messages([
            ("system",
            """You are an AI specialist in dynamic content categorization and titling. Your task is to update chunk titles when new information is added, ensuring they remain accurate and informative.

            Guidelines for updating chunk titles:
            1. Analyze all contexts in the chunk, including the newly added one.
            2. Consider the current chunk summary and title.
            3. Create a title that encompasses all main themes in the chunk.
            4. Be extremely concise (less than 20 words).
            5. Use generalizations where appropriate (e.g., "Fruit" for various specific fruits).
            6. Maintain specificity only when crucial to the chunk's identity.
            7. Use title case (capitalize main words).
            8. Employ ampersands (&) instead of "and" to save space.
            9. Ensure the new title distinguishes this chunk from others.
            10. Avoid drastic changes unless the new context significantly alters the chunk's focus.

            Title creation principles:
            - Brevity: Aim for the shortest title that still captures the essence.
            - Clarity: Use clear, accessible language.
            - Comprehensiveness: Reflect all major themes in the chunk.
            - Distinctiveness: Ensure the title uniquely identifies this chunk.

            Examples:
            Input:
            Contexts:
                - Apples are rich in fiber.
                - Bananas contain high levels of potassium.
                - New: Oranges are an excellent source of vitamin C.
            Current Summary: Nutritional benefits of common fruits.
            Current Title: Apple & Banana Nutrition

            Output: Fruit Nutritional Profiles

            Respond only with the new chunk title, without any additional text or explanations."""),
            ("user", """Based on the following information, generate an updated title for the content chunk:

            Contexts:
            {context}

            Current Summary: {current_summary}
            Current Title: {current_title}"""),
        ])

        runnable = PROMPT | self.llm

        updated_chunk_title = runnable.invoke({
            "context": "\n".join(chunk['contexts']),
            "current_summary": chunk['summary'],
            "current_title": chunk['title']
        }).content

        return updated_chunk_title

    def _get_new_chunk_summary(self, context):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", 
            """You are an AI assistant specializing in content summarization and organization. Your task is to create concise, informative summaries for content chunks based on their contexts.

            Guidelines for generating chunk summaries:
            1. Be brief yet comprehensive (aim for 10-15 words).
            2. Capture the main topic, theme, and key points of the chunk.
            3. Use generalizations where appropriate, but retain specificity when crucial.
            4. Align with existing information while highlighting new or unique content.
            5. Use clear, accessible language avoiding jargon unless necessary.
            6. Ensure the summary distinguishes this chunk from others.
            7. Include relevant dates, names, or numerical data if they are central to the content.
            8. Use present tense and active voice where possible.
            9. Avoid personal opinions or interpretations; stick to the facts presented.
            10. If the chunk contains multiple topics, prioritize the most significant ones.

            Examples:
            Context: Detailed analysis of climate change effects on polar bear populations in the Arctic from 2000 to 2020.
            Summary: Climate Change Impact on Arctic Polar Bears (2000-2020): Population Trends and Habitat Loss

            Context: Comparison of machine learning algorithms for natural language processing tasks, focusing on BERT, GPT, and LSTM.
            Summary: NLP Algorithm Comparison: BERT, GPT, and LSTM Performance in Various Language Tasks

            Your response should contain only the new chunk summary, without any additional text or explanations."""),
            ("user", "Based on the following context, generate an appropriate summary for the content chunk:\n\n{context}"),
        ])

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "context": context
        }).content

        return new_chunk_summary
    
    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system",
            """You are an AI specialist in content categorization and titling. Your task is to create concise, informative titles for content chunks based on their summaries.

            Guidelines for generating chunk titles:
            1. Be extremely brief (3-5 words maximum).
            2. Capture the core theme or main topic of the chunk.
            3. Use generalizations where appropriate, but maintain specificity if crucial.
            4. Ensure the title is distinct from other potential chunk titles.
            5. Use title case (capitalize main words).
            6. Avoid articles (a, an, the) unless absolutely necessary.
            7. Use ampersands (&) instead of "and" to save space.
            8. Include key terms or concepts that define the chunk's content.
            9. If applicable, use broader categories (e.g., "Fruit" instead of specific fruits).
            10. For time-related content, use general terms like "Chronology" or "Timeline".

            Examples:
            Summary: This chunk discusses various fruits, their nutritional values, and health benefits.
            Title: Fruit Nutrition & Benefits

            Summary: The passage covers the history of computing from the 1950s to the present day.
            Title: Computing History Timeline

            Summary: This section compares different machine learning algorithms for image recognition tasks.
            Title: ML for Image Recognition

            Respond only with the new chunk title, without any additional text or explanations."""),
            ("user", "Based on the following summary, generate an appropriate title for the content chunk:\n\n{summary}"),
        ])

        runnable = PROMPT | self.llm

        new_chunk_title = runnable.invoke({
            "summary": summary
        }).content

        return new_chunk_title


    def _create_new_chunk(self, context):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit] # I don't want long ids
        new_chunk_summary = self._get_new_chunk_summary(context)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'contexts': [context],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks)
        }
        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")
    
    def get_chunk_outline(self):
        """
        Get a string which represents the chunks you currently have.
        This will be empty when you first start off
        """
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ID: {chunk['chunk_id']}\nChunk Name: {chunk['title']}\nChunk Summary: {chunk['summary']}\n\n"""
        
            chunk_outline += single_chunk_string
        
        return chunk_outline

    def _find_relevant_chunk(self, context):
        current_chunk_outline = self.get_chunk_outline()

        PROMPT = ChatPromptTemplate.from_messages([
            ("system",
            """You are an AI expert in content analysis and categorization. Your task is to determine whether a given context belongs to any existing content chunks based on semantic similarity and thematic relevance.

            Guidelines for matching context to chunks:
            1. Analyze the semantic meaning, direction, and intention of the context.
            2. Compare the context against each existing chunk's name and summary.
            3. Look for thematic overlap, shared concepts, or logical connections.
            4. Consider broader categories and potential implicit relationships.
            5. Prioritize strong matches over weak or tangential connections.
            6. Be consistent in your matching criteria across different contexts.
            7. Consider the level of specificity in both the context and the chunks.
            8. If a context could fit multiple chunks, choose the most relevant one.
            9. Avoid forcing a match if there isn't a clear thematic connection.
            10. Remember that it's okay for a context to not match any existing chunks.

            Response format:
            - If you find a matching chunk, return only the Chunk ID (e.g., "2n4l3d").
            - If no suitable chunk is found, return exactly "No chunks".

            Example:
            Input:
                Context: "The Golden Gate Bridge was painted its iconic orange color in 1933."
                Current Chunks:
                    - Chunk ID: 2n4l3d
                    - Chunk Name: San Francisco Landmarks
                    - Chunk Summary: Overview of famous structures and locations in San Francisco

                    - Chunk ID: 93833k
                    - Chunk Name: History of Bridge Construction
                    - Chunk Summary: Timeline and techniques of major bridge constructions worldwide
            Output: 2n4l3d

            Remember, only respond with a Chunk ID or "No chunks". Provide no additional explanation."""),
            ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--\n\nDetermine if the following context should belong to one of the chunks outlined:\n{context}"),
        ])

        runnable = PROMPT | self.llm

        chunk_found = runnable.invoke({
            "context": context,
            "current_chunk_outline": current_chunk_outline
        }).content

        if len(chunk_found) != self.id_truncate_limit:
            return None

        return chunk_found
    
    def get_chunks(self, get_type='dict'):
        """
        This function returns the chunks in the format specified by the 'get_type' parameter.
        If 'get_type' is 'dict', it returns the chunks as a dictionary.
        If 'get_type' is 'list_of_strings', it returns the chunks as a list of strings, where each string is a proposition in the chunk.
        """
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk['contexts']]))
            return chunks
    
    def pretty_print_chunks(self):
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print(f"Contexts:")
            for prop in chunk['contexts']:
                print(f"    -{prop}")
            print("\n\n")

    def pretty_print_chunk_outline(self):
        print("Chunk Outline\n")
        print(self.get_chunk_outline())

if __name__ == "__main__":
    ac = AgenticChunker()

    ## Comment and uncomment the propositions to your hearts content
    propositions = [
        'The month is October.'
        'The year is 2023.',
        "One of the most important things that I didn't understand about the world as a child was the degree to which the returns for performance are superlinear.",
        'Teachers and coaches implicitly told us that the returns were linear.',
        "I heard a thousand times that 'You get out what you put in.'",
        'Teachers and coaches meant well.',
        "The statement that 'You get out what you put in' is rarely true.",
        "If your product is only half as good as your competitor's product, you do not get half as many customers.",
        "You get no customers if your product is only half as good as your competitor's product.",
        'You go out of business if you get no customers.',
        'The returns for performance are superlinear in business.',
        'Some people think the superlinear returns for performance are a flaw of capitalism.',
        'Some people think that changing the rules of capitalism would stop the superlinear returns for performance from being true.',
        'Superlinear returns for performance are a feature of the world.',
        'Superlinear returns for performance are not an artifact of rules that humans have invented.',
        'The same pattern of superlinear returns is observed in fame.',
        'The same pattern of superlinear returns is observed in power.',
        'The same pattern of superlinear returns is observed in military victories.',
        'The same pattern of superlinear returns is observed in knowledge.',
        'The same pattern of superlinear returns is observed in benefit to humanity.',
        'In fame, power, military victories, knowledge, and benefit to humanity, the rich get richer.'
    ]
    
    ac.add_contexts(propositions)
    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    print(ac.get_chunks(get_type='list_of_strings'))