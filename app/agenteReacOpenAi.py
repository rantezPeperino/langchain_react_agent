import requests
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_react_agent
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

# Asegúrate de reemplazar 'your-openai-api-key' con tu clave real de OpenAI
openai_api_key = ''

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

@tool
def get_user_info(name: str) -> str:
    """Obtiene la información de un usuario basado en su nombre."""
    response = requests.get('https://jsonplaceholder.typicode.com/users')
    users = response.json()
    
    # for user in users:
    #     if user['name'].lower() == name.lower():
    #         return str(user)
    
    return users #"Usuario no encontrado." se comenta para que el llm busque en toda la lista.

# Crear el agente REACT
tools = [get_user_info]

template = """Responde las siguientes preguntas lo mejor que puedas y en español. Tienes acceso a las siguientes herramientas:

{tools}

Utilice el siguiente formato:

Question: La pregunta de entrada que debes responder
Thought: Siempre debes pensar en qué hacer
Action: La acción a tomar debe ser una de [{tool_names}]
Action Input: La entrada a la acción
Observation: El resultado de la acción
... (este Pensamiento/Acción/Entrada de Acción/Observación puede repetirse N veces)
Thought: Ahora sé la respuesta final.
Final Answer: La respuesta final a la pregunta de entrada original

Comenzar!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

# Función para invocar el agente
def query_agent(query: str):
    return agent_executor.invoke({"input": query})

# Ejemplo de uso
result = query_agent("Obtén la email y el codigo postal del usuario Leanne Graham")
#result = query_agent("cual es el telefono y mail de Chelsey Dietrich")
print(result)