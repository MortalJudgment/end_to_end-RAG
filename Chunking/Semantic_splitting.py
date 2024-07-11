import re
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    def __init__(self, model_name='BAAI/bge-small-en-v1.5'):
        self.model = SentenceTransformer(model_name)

    def split_into_sentences(self, text):
        lines = text.split('\n')
        sentences = []
        for line in lines:
            line = line.strip()
            if line:
                if re.match(r'^#+\s', line):
                    sentences.append(line)
                else:
                    line_sentences = re.split(r'(?<=[.!?])\s+', line)
                    sentences.extend(line_sentences)
        return sentences

    def combine_sentences(self, sentences, buffer_size=1):
        for i in range(len(sentences)):
            combined_sentence = ''
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]['sentence'] + ' '
            combined_sentence += sentences[i]['sentence']
            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + sentences[j]['sentence']
            sentences[i]['combined_sentence'] = combined_sentence
        return sentences

    def generate_embeddings(self, sentences):
        embeddings = self.model.encode([x['combined_sentence'] for x in sentences], batch_size=16, show_progress_bar=True, normalize_embeddings=True)
        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]
        return sentences

    def calculate_cosine_distances(self, sentences):
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            distance = 1 - similarity
            distances.append(distance)
            sentences[i]['distance_to_next'] = distance
        return distances, sentences

    def visualize_distances(self, distances, breakpoint_percentile_threshold=95):
        plt.plot(distances)
        max_distance = max(distances)
        y_upper_bound = round(max_distance * 2) / 2
        plt.ylim(0, y_upper_bound)
        plt.xlim(0, len(distances))
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-')
        num_distances_above_threshold = len([x for x in distances if x > breakpoint_distance_threshold])
        plt.text(x=(len(distances) * .01), y=y_upper_bound / 50, s=f"{num_distances_above_threshold + 1} Chunks")
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, breakpoint_index in enumerate(indices_above_thresh):
            start_index = 0 if i == 0 else indices_above_thresh[i - 1]
            end_index = breakpoint_index if i < len(indices_above_thresh) - 1 else len(distances)
            plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
            plt.text(x=np.average([start_index, end_index]),
                     y=breakpoint_distance_threshold + (y_upper_bound) / 20,
                     s=f"Chunk #{i}", horizontalalignment='center',
                     rotation='vertical')
        if indices_above_thresh:
            last_breakpoint = indices_above_thresh[-1]
            if last_breakpoint < len(distances):
                plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(indices_above_thresh) % len(colors)], alpha=0.25)
                plt.text(x=np.average([last_breakpoint, len(distances)]),
                         y=breakpoint_distance_threshold + (y_upper_bound) / 20,
                         s=f"Chunk #{i + 1}",
                         rotation='vertical')
        plt.title("PG Essay Chunks Based On Embedding Breakpoints")
        plt.xlabel("Index of sentences in essay (Sentence Position)")
        plt.ylabel("Cosine distance between sequential sentences")
        plt.show()

    def chunk_text(self, text, buffer_size=1, breakpoint_percentile_threshold=95):
        sentences = [{'sentence': sentence} for sentence in self.split_into_sentences(text)]
        sentences = self.combine_sentences(sentences, buffer_size)
        sentences = self.generate_embeddings(sentences)
        distances, sentences = self.calculate_cosine_distances(sentences)
        self.visualize_distances(distances, breakpoint_percentile_threshold)
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
        start_index = 0
        chunks = []
        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = index + 1
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)
        return chunks

if __name__ == "__main__":
    chunker = SemanticChunker()
    text = """
    # LLM Agents: Enhancing AI Capabilities

    ## Introduction to LLM Agents

    Large Language Model (LLM) Agents represent a significant advancement in artificial intelligence, combining the power of language models with autonomous decision-making capabilities. These agents leverage the vast knowledge and language understanding of LLMs to interact with their environment, make decisions, and perform tasks with minimal human intervention.

    ## Key Components of LLM Agents

    ### 1. Large Language Model Core

    At the heart of an LLM Agent is a powerful language model such as GPT-4, PaLM, or Claude. This core provides:

    - Natural language understanding and generation
    - Vast knowledge across diverse domains
    - Ability to process and respond to complex queries

    ### 2. Action Space

    The action space defines the set of operations an LLM Agent can perform. This may include:

    - Web searches
    - API calls
    - Database queries
    - File system operations
    - External tool interactions

    Example,
    ```python
    import requests

    # Example of a function for making a web search
    def web_search(query):
        response = requests.get(f"https://api.duckduckgo.com/?q=\{query\}&format=json")
        results = response.json()["RelatedTopics"]
        return results[:3]  # Return top 3 results

    # Example of a function to call a weather API
    def get_weather(city):
        response = requests.get(f"http://api.weatherapi.com/v1/current.json?key=your_api_key&q=\{city\}")
        return response.json()["current"]

    # Example usage
    print(web_search("OpenAI"))
    print(get_weather("London"))
    ```

    ### 3. Planning and Reasoning Module

    This component enables the agent to:

    - Break down complex tasks into manageable steps
    - Formulate strategies to achieve goals
    - Adapt plans based on new information or obstacles

    Example,
    ```python
    class PlanningModule:
        def break_down_task(self, task):
            # Simplified task breakdown
            steps = [
                f"Step 1: Analyze {task}",
                f"Step 2: Research {task}",
                f"Step 3: Implement solution for {task}",
                f"Step 4: Test and validate {task}"
            ]
            return steps

        def adapt_plan(self, original_plan, new_information):
            # Simplified plan adaptation
            adapted_plan = original_plan.copy()
            adapted_plan.append(f"Additional step: Incorporate {new_information}")
            return adapted_plan

    planner = PlanningModule()
    task = "Develop an AI chatbot"
    initial_plan = planner.break_down_task(task)
    adapted_plan = planner.adapt_plan(initial_plan, "Natural Language Processing")
    ```

    ### 4. Memory Systems

    LLM Agents often incorporate various memory types:

    - Short-term memory: For maintaining context within a conversation or task
    - Long-term memory: For storing and retrieving information across multiple sessions
    - Episodic memory: For learning from past experiences and improving performance over time

    Example,
    ```python
    class MemorySystem:
        def __init__(self):
            self.short_term_memory = []
            self.long_term_memory = {}
            self.episodic_memory = []

        def add_to_short_term(self, item):
            self.short_term_memory.append(item)
            if len(self.short_term_memory) > 5:  # Limit short-term memory
                self.short_term_memory.pop(0)

        def add_to_long_term(self, key, value):
            self.long_term_memory[key] = value

        def add_to_episodic(self, episode):
            self.episodic_memory.append(episode)

        def retrieve_from_long_term(self, key):
            return self.long_term_memory.get(key, "Not found")

    memory = MemorySystem()
    memory.add_to_short_term("Recent user query")
    memory.add_to_long_term("user_preference", "Dark mode")
    memory.add_to_episodic({"task": "Web search", "result": "Successful"})
    ```

    ### 5. Perception Module

    This allows the agent to process and understand various input types:

    - Text input
    - Structured data
    - Images (in multimodal systems)
    - Audio (in speech-enabled agents)

    Example,
    ```python
    import json
    from PIL import Image

    class PerceptionModule:
        def process_text(self, text):
            return f"Processed text: {text}"

        def process_structured_data(self, data):
            return json.loads(data)

        def process_image(self, image_path):
            image = Image.open(image_path)
            # Simplified image processing
            return f"Processed image of size {image.size}"

    perception = PerceptionModule()
    text_result = perception.process_text("Hello, AI!")
    json_result = perception.process_structured_data('{"name": "AI Agent", "type": "LLM"}')
    image_result = perception.process_image("example.jpg")
    ```

    ### 6. Output Generation and Refinement

    Responsible for producing coherent and appropriate responses, this module may include:

    - Response filtering for safety and relevance
    - Style adaptation based on user preferences or task requirements
    - Multi-turn response generation for complex interactions

    Example,
    ```
    class OutputModule:
        def __init__(self, safety_filter, style_adapter):
            self.safety_filter = safety_filter
            self.style_adapter = style_adapter

        def generate_response(self, content):
            filtered_content = self.safety_filter(content)
            styled_content = self.style_adapter(filtered_content)
            return styled_content

        def multi_turn_generation(self, conversation):
            response = ""
            for turn in conversation:
                response += self.generate_response(turn) + " "
            return response.strip()

    def simple_safety_filter(content):
        # Simplified safety filter
        unsafe_words = ["unsafe", "harmful"]
        for word in unsafe_words:
            content = content.replace(word, "[FILTERED]")
        return content

    def simple_style_adapter(content):
        # Simplified style adapter
        return content.upper()

    output_module = OutputModule(simple_safety_filter, simple_style_adapter)
    response = output_module.generate_response("This is a safe response.")
    multi_turn = output_module.multi_turn_generation(["Hello", "How are you?"])
    ```

    ## Applications of LLM Agents

    LLM Agents find applications in numerous fields:

    1. **Customer Service**: Handling complex customer queries and providing personalized assistance
    2. **Research and Analysis**: Conducting literature reviews, data analysis, and generating reports
    3. **Personal Assistants**: Managing schedules, answering questions, and performing tasks on behalf of users
    4. **Education**: Providing personalized tutoring and answering student questions across various subjects
    5. **Healthcare**: Assisting in diagnosis, treatment planning, and patient education
    6. **Software Development**: Helping with code generation, debugging, and documentation
    7. **Creative Writing**: Assisting authors with story development, character creation, and editing

    ## Challenges and Considerations

    Despite their potential, LLM Agents face several challenges:

    ### Ethical Considerations

    - Ensuring privacy and data security
    - Preventing biased or discriminatory outputs
    - Maintaining transparency about AI involvement in interactions

    ### Technical Challenges

    - Handling ambiguity and uncertainty in complex scenarios
    - Maintaining coherence and consistency across long interactions
    - Integrating with external systems and APIs securely and efficiently

    ### Performance and Scalability

    - Optimizing response times for real-time applications
    - Managing computational resources for large-scale deployments
    - Balancing accuracy with efficiency in resource-constrained environments

    ## Future Directions

    The field of LLM Agents is rapidly evolving, with several exciting areas of development:

    1. **Multimodal Agents**: Integrating vision, speech, and other sensory inputs for more comprehensive understanding and interaction
    2. **Collaborative Agents**: Developing systems where multiple agents work together to solve complex problems
    3. **Continual Learning**: Implementing mechanisms for agents to learn and improve from ongoing interactions without full retraining
    4. **Explainable AI**: Enhancing the ability of agents to provide clear reasoning for their decisions and actions
    5. **Domain-Specific Agents**: Creating highly specialized agents for fields like law, medicine, or engineering
    6. **Emotional Intelligence**: Developing agents capable of recognizing and responding to human emotions effectively

    ## Best Practices for Implementing LLM Agents

    When developing and deploying LLM Agents, consider the following best practices:

    1. **Clear Scope Definition**: Clearly define the agent's capabilities and limitations to manage user expectations
    2. **Robust Testing**: Implement comprehensive testing protocols, including edge cases and potential failure modes
    3. **User Feedback Integration**: Establish mechanisms to collect and incorporate user feedback for continuous improvement
    4. **Ethical Guidelines**: Develop and adhere to strict ethical guidelines for agent behavior and decision-making
    5. **Fallback Mechanisms**: Implement graceful degradation and human handoff options for situations beyond the agent's capabilities
    6. **Performance Monitoring**: Set up robust monitoring systems to track agent performance, detect anomalies, and ensure reliability
    7. **Version Control**: Maintain careful version control of agent models and configurations to manage updates and rollbacks effectively

    ## Conclusion

    LLM Agents represent a powerful fusion of natural language processing, decision-making algorithms, and task automation. As these systems continue to evolve, they promise to revolutionize how we interact with AI, potentially transforming industries and opening new frontiers in human-AI collaboration. However, their development and deployment must be guided by careful consideration of ethical implications, technical challenges, and the need for responsible AI practices.
    """
    chunks = chunker.chunk_text(text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk #{i}\n{chunk}\n")