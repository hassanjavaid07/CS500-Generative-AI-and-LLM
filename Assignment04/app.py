"""
###<u> **GENERATIVE AI & LLM PROGRAMMING ASSIGNMENT # 4** </u>
* **NAME = HASSAN JAVAID**
* **ROLL NO. = MSCS23001**
* **TASK = Implementation of Multi-Agentic Retreival Augmented Generation (RAG) for document and search related queries**
* **LLM used: CHATGROQ WITH RAG**

This file i.e. app.py is shared for deployment on Hugging Face Spaces. This file was submitted as part of course
CS-500 Generative AI & LLM conducted in ITU, Lahore during Fall-2024.g

Hugging Face Space Link: https://huggingface.co/spaces/trident-10/Researcher-RAG/tree/main


GitHub Repo Link: https://github.com/hassanjavaid07/CS500-Generative-AI-and-LLM/tree/main/Assignment04


This file and relavant repos are the property of the author and is under MIT License. Give credit when sharing.
"""


import os
import asyncio
import dotenv
import gradio as gr
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import MessagesState
from langchain.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
import pinecone
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from typing_extensions import TypedDict
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Load environment variables
dotenv.load_dotenv()

# Initialize Pinecone with API key and environment
pc = pinecone.Pinecone(
    api_key=os.environ['PINECONE_API_KEY'],  
    environment=os.environ.get('PINECONE_ENVIRONMENT')  
)

index_name = "gen-ai-hw4"

# Ensure the index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embedding model
        metric='cosine'  
    )
index = pc.Index(index_name)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embedding_model,  
    text_key="text"  
)


SYS_PROMPT = """

Based on the content of your PDF document, here's a prompt to gather information:

"Gather information from the Netsol investor relations report PDF document. Please extract the following data points:

1. Financial Highlights:
	* Revenue figures for the past two years
	* Net income figures for the past two years
	* Gross profit margin percentages for the past two years
	* Total assets and liabilities figures for the past two years
2. Board of Directors and Senior Management:
	* Names and positions of the company's board of directors
	* Names and positions of the company's senior management team (including the Chairman, CEO, CFO, etc.)
3. Company Profile:
	* Overview of the company's products/services
	* Main business segments
	* Mission and vision statements
	* Brief history of the company
4. Visualizations and Graphs:
	* Identify any graphs or charts that show trends in revenue, net income, or other key financial metrics
	* Extract any information from infographics or plots that provide insights into the company's performance or industry trends
5. Financial Terms:
	* Define and provide examples of key financial terms used throughout the report (e.g., EBITDA, ROCE, etc.)
6. Images and Pictures:
	* Identify the names and roles of the company's board of directors and senior management team mentioned in the report
	* Describe any notable events or milestones mentioned in the report

Please organize the extracted information into clear and concise sections, and provide any additional context or clarifications where necessary."

"""

# Agnetic Tools Definition
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b.

    Args:
        a: first int
        b: second int

    Returns:
        The sum of a and b.
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a.

    Args:
        a: first int
        b: second int

    Returns:
        The difference of a and b.
    """
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: numerator
        b: denominator

    Returns:
        The division of a by b.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

@tool
def exponentiate(a: int, b: int) -> int:
    """Raise a to the power of b.

    Args:
        a: base
        b: exponent

    Returns:
        a raised to the power of b.
    """
    return a ** b

# Tavily search tool
@tool
def search_tool(query: str, max_results: int = 3) -> str:
    """
    Perform a search query using the Tavily search tool to retrieve information.

    This function utilizes the Tavily search tool to perform a web search
    for the given query and returns the results. It is useful for answering
    questions or retrieving information from the web.

    Args:
        query: The search query string to be executed.
        max_results: The maximum number of search results
            to retrieve. Defaults to 3.

    Returns:
        str: A string containing the search results. If an error occurs during
            the search, an error message is returned instead.

    Raises:
        Exception: If there is an issue with the Tavily search tool invocation.

    Example:
        >>> search_tool("Who won the last match between Pakistan and Zimbabwe?")
        'Pakistan won the last match by 5 wickets.'
    """
    print("In search")
    tavily_search = TavilySearchResults(max_results=max_results)
    try:
        return tavily_search.invoke(query)
    except Exception as e:
        return f"Error performing search: {e}"



# Fetches document score
def scoreDocuments(docs, query, embedding_model, threshold=0.7):
    """
    Scores the relevance of documents to the query using cosine similarity.

    Args:
        docs: List of retrieved documents.
        query: The user query.
        embedding_model: Instance of HuggingFaceEmbeddings for generating embeddings.
        threshold: Minimum relevance score to consider documents relevant.

    Returns:
        bool: Whether the documents are relevant based on the threshold.
        list: List of relevance scores.
    """
    # Generate embedding for the query
    query_embedding = embedding_model.embed_query(query)

    # Generate embeddings for each document
    doc_embeddings = [embedding_model.embed_query(doc.page_content) for doc in docs]

    # Compute cosine similarity scores
    scores = [cosine_similarity([query_embedding], [doc_embedding])[0][0] for doc_embedding in doc_embeddings]

    # Check if all scores meet the relevance threshold
    is_relevant = all(score >= threshold for score in scores)
    return is_relevant, scores


# Augments the prompt
def augmentPrompt(context: str, query: str) -> str:
    """
    Combines the system-level prompt with the user's query and the relevant document context.

    Args:
        context: The retrieved document context for the query.
        query: The user's original query.

    Returns:
        str: The full prompt for the LLM, including system instructions and query context.
    """
    prompt = f"""
    {SYS_PROMPT}

    The user asked: {query}

    The relevant context is:

    {context}
    """

    return prompt


# Tool Definition
@tool
def doc_query_tool(query: str):
    """
    Fetches relevant context from Pinecone, scores relevance, and handles query refinement if needed.
    Invokes the Groq LLM for generating responses.

    Args:
        query: The user's query.

    Returns:
        str: The response generated by the LLM based on the provided or refined query.
    """
    print("In doc_query")
    # Retrieve relevant documents using LangChain's Pinecone integration
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(query)

    # Score documents for relevance
    is_relevant, scores = scoreDocuments(retrieved_docs, query, embedding_model, threshold=0.5)
    if is_relevant:
        print("In is_relevant")
        # Generate prompt with relevant context
        context = ''.join(f'## Chunk {i}:\n\n{doc.page_content}\n\n' for i, doc in enumerate(retrieved_docs))
        prompt = augmentPrompt(context, query)
        response = llm.invoke([HumanMessage(content=prompt)])
        # return {"messages": [response]}
        if context:
            print(f"context = {context}")
        return response

    else:
        # Rewrite the query using the LLM
        print("In query rewrite")
        chat_model = ChatGroq(model="llama3-8b-8192", api_key=os.environ["GROQ_API_KEY"])
        rewrite_msg = [
            HumanMessage(
                content=f""" \n
                Look at the input and try to reason about the underlying semantic intent/meaning. \n
                Here is the initial question:
                \n ------- \n
                {query}
                \n ------- \n
                Formulate an improved question: """,
            )
        ]
        rewritten_query = chat_model.invoke(rewrite_msg)

        # # Fetch documents again with the rewritten query
        new_retrieved_docs = retriever.get_relevant_documents(rewritten_query.content)

        # Generate prompt with the new context
        new_context = ''.join(f'## Chunk {i}:\n\n{doc.page_content}\n\n' for i, doc in enumerate(new_retrieved_docs))
        new_prompt = augmentPrompt(new_context, rewritten_query.content)
        response = llm.invoke([HumanMessage(content=new_prompt)])
        if new_context:
            print(f"new_context = {new_context}")
        return response

@tool
def general_answer_tool(query: str):
    """Tool for handling non-specific queries (e.g., facts or definitions)."""
    print("In general")
    # query = state["messages"][-1].content.lower()
    response = llm.invoke([HumanMessage(content=f"Answer the following general query: {query}")])
    return response


# LangGraph and Nodes/Agents Setup
members = ['doc_query', 'tavilysearch', 'general']
options = members + ["FINISH"]

system_prompt = """
You are a supervisor tasked with managing a conversation between the following workers: {members}.
Given the following user request, respond with the worker to act next.
Each worker will perform a task and respond with their results and status.
When finished, respond with FINISH.
"""


class Router(TypedDict):
    next: Literal['doc_query', 'tavilysearch', 'general', "FINISH"]

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", api_key=os.environ['GROQ_API_KEY'])

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=options, members=", ".join(members))


# Supervisor Node Setup
def supervisor_node(state: MessagesState) -> Command[Literal['doc_query', 'tavilysearch', 'general', "__end__"]]:
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    # print(messages)
    response = llm.with_structured_output(Router).invoke(messages)
    # print(response)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)

# Agents Setup
# Math Agent
# math_prompt = "Peform arithmetic operations using your given tools"
math_agent = create_react_agent(llm, 
                                tools=[multiply, add, subtract, divide, exponentiate],
                                state_modifier="You will ONLY DO math.")

def math_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = math_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="math")]},
        goto="supervisor",
    )

# Search Agent
search_agent = create_react_agent(llm, 
                                  tools=[search_tool],
                                  state_modifier="You are a researcher. DO NOT do any math.")

def search_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="tavilysearch")]},
        goto="supervisor",
    )

# Document Query Agent
doc_query_agent = create_react_agent(llm, 
                                     tools=[doc_query_tool],
                                     state_modifier="You will only look into retreived documents for answer. DO NOT search on internet.")

def doc_query_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = doc_query_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="doc_query")]},
        goto="supervisor",
    )

# # General Answer Agent
general_agent = create_react_agent(llm, 
                                   tools=[general_answer_tool],
                                   state_modifier="You will ONLY GIVE answer to the query if no else tool can give an answer. DO NOT do math.")

def general_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    print("In general_node")
    # print(state)
    result = general_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="general")]},
        goto="supervisor",
    )

# Build the StateGraph
builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
# builder.add_node("math", math_node)
builder.add_node("tavilysearch", search_node)
builder.add_node("doc_query", doc_query_node)
builder.add_node("general", general_node)
# builder.add_edge("supervisor", END)
graph = builder.compile()


# Gardio App Creation
def convertQueryToInputsFormat(query):
    return {"messages": [('human', query)]}


async def getFinalGraphResponse(graph, inputs, stream_mode="values"):
    final_chunk = None
    async for chunk in graph.astream(inputs, stream_mode=stream_mode):
        final_chunk = chunk  
    return final_chunk

def getResponse(input_text):
    inputs = convertQueryToInputsFormat(input_text)
    try:
        loop = asyncio.get_event_loop()
    # Handle cases where no loop exists
    except RuntimeError:  
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    final_output = loop.run_until_complete(getFinalGraphResponse(graph, inputs))

    if final_output and "messages" in final_output:
        response = final_output["messages"][-1].content
        return response
    else:
        return "No response received."

# Create the Gradio Interface
iface = gr.Interface(
    fn=getResponse, 
    inputs=gr.Textbox(
        label="Enter your question",
        placeholder="Type your question here..."
    ), 
    outputs="textbox",  
    title="Researcher and Doc-Query Handler",
    description=(
        "Ask a question about NetSol Financial Report or internet related query "
        "This assistant looks up relevant documents if needed and then answers your question."
    ),
    examples=[
        ["What are the main objectives outlined in NETSOL's mission statement?"], 
        ["Who won first t20 match between Pakistan and Zimbabwe?"],
        ["Who is the CEO of Huawei?"]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

iface.launch(share=False)  
