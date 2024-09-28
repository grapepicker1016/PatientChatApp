import re
from django.shortcuts import render, redirect
from django.http import JsonResponse

from django.contrib import auth
from django.contrib.auth.models import User
from django.db.models import Q
from .models import Chat, Patient, Reservation, Doctor, ReservationRequest

from django.utils import timezone, dateparse

import os
from typing import Dict, Optional, Sequence


from pymongo.database import Database as MongoDatabase

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from typing import Annotated, List, Dict, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import operator
from typing import Annotated, Sequence, TypedDict
from datetime import date, time

from langchain_openai import ChatOpenAI
import functools

from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode, InjectedState
from typing import Literal
from IPython.display import Image, display
from langchain_community.utilities import SerpAPIWrapper

from exa_py import Exa
import json
from py2neo import Graph, Node

import psycopg2
from dateutil import parser


def change_date_format(date_string):
    try:

        # Parse the input date string
        parsed_date = parser.parse(date_string, dayfirst=True)

        # Format the date object to the desired output format
        new_date_string = parsed_date.strftime("%Y/%m/%d")

        return new_date_string
    except ValueError:
        return "Invalid date format"


os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"
os.environ["SERPAPI_API_KEY"] = "YOUR_SERPAPI_API_KEY"
os.environ["EXA_API_KEY"] = "YOUR_EXA_API_KEY"

# PostgreSQL database connection

conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="postgres",
    user="postgres",
    password="password"
)

cur = conn.cursor()

# # Create table
cur.execute("""
    CREATE TABLE IF NOT EXISTS Insight(
        id SERIAL PRIMARY KEY,
        username VARCHAR(100),
        content TEXT
    )
""")

# conn.commit()
cur.close()
conn.close()

# Create


def get_connection():
    return psycopg2.connect(
        host="localhost",
        port="5432",
        database="postgres",
        user="postgres",
        password="password"
    )


def create_insight(username, content):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO Insight (username, content) VALUES (%s, %s)", (username, content))
    conn.commit()
    cur.close()
    conn.close()

# Read


def get_insight(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Insight WHERE username = %s", (username,))
    insight = cur.fetchone()
    cur.close()
    conn.close()
    return insight

# Update


def update_insight(username, new_content):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE Insight SET content = %s WHERE username = %s",
                (new_content, username))
    conn.commit()
    cur.close()
    conn.close()

# Delete


def delete_insight(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM Insight WHERE username = %s", (username,))
    conn.commit()
    cur.close()
    conn.close()


serp_api_key = os.getenv("SERPAPI_API_KEY")

# Graph database connection

# Replace these with your Neo4j credentials
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DB_NAME = "hospitalagent"

# Create connection

try:
    # Connect to the Neo4j graph database
    graphdb = Graph(NEO4J_URI, auth=(
        NEO4J_USERNAME, NEO4J_PASSWORD), name=NEO4J_DB_NAME)
    print(f"Connected to Neo4j Successfully")
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")
    graphdb = None

# Initialize Neo4j Database

# Patient nodes

# patients = Patient.objects.filter()
# for patient in patients:
#     patientNode = Node(
#         "Patient",
#         username=patient.user.username,
#         name=f"{patient.first_name} {patient.last_name}",
#         phaone_number=patient.phone_number,
#         location=patient.location,
#         date_of_birth=patient.date_of_birth,
#         gender=patient.gender,
#         medical_condition=patient.medical_condition,
#         medical_regimen=patient.medical_regimen,
#         diet=patient.diet,
#         created_at=str(patient.created_at)
#     )
#     graphdb.create(patientNode)

# # Doctor nodes
# doctors = Doctor.objects.filter()
# for doctor in doctors:
#     doctorNode = Node(
#         "Doctor",
#         username=doctor.user.username,
#         name=f"{doctor.first_name} {doctor.last_name}",
#         phaone_number=doctor.phone_number,
#         major=doctor.major,
#         created_at=str(patient.created_at)
#     )
#     graphdb.create(doctorNode)

# # Reservation relationship
# reservations = Reservation.objects.filter()
# for reservation in reservations:
#     patientUsername = reservation.patient.user.username
#     doctorUsername = reservation.doctor.user.username
#     # query = "MATCH (p:Patient), (d:Doctor) WHERE p.user.username = " + patientUsername + " AND d.user.username = " + doctorUsername + " CREATE (p)-[:reservation]->(d) RETURN p, d"
#     query = """
#         MATCH (p:Patient), (d:Doctor)
#         WHERE p.username = $patientUsername AND d.username = $doctorUsername
#         CREATE (p)-[:reservation]->(d)
#         RETURN p, d
#         """

#     graphdb.run(query, patientUsername=patientUsername,
#                 doctorUsername=doctorUsername)

llm = ChatOpenAI(model="gpt-4o")  # Customizable

# Create Graph

# Define State


def format_text(text):
    # Replace ### with <h3>
    text = re.sub(r'###\s*(.+?):', r'<h3><br>\1:</h3>', text)

    # Replace **text** with <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Replace - with <br> for line breaks
    text = re.sub(r'-\s', r'<br>', text)

    # Replace [text](url) with <a href="url">text</a>
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # Ensure links are displayed as list items
    text = re.sub(r'(<a href="[^"]+">[^<]+</a>)', r'<li>\1</li>', text)

    # Wrap the list items in <ul> tags
    text = re.sub(r'(<li>.*?</li>)', r'<ul><br>\1</ul>', text, flags=re.DOTALL)

    return text


class HospitalAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    patient_username: str
    response: str

# Define Tools


@ tool("get_patient_detail")
def get_patient_detail(
    state: Annotated[dict, InjectedState]
) -> Patient:
    """Tool to fetch patient detail"""
    # return state["messages"][-1].content
    # return "User chat history"
    try:
        user = User.objects.get(username=state["patient_username"])
        patient = Patient.objects.get(user=user)
        return patient
    except User.DoesNotExist:
        return None


@ tool("get_doctor_detail")
def get_doctor_detail(
    doctor_name: Annotated[str, "Doctor's name"]
) -> Doctor:
    """Tool to fetch information of a doctor from his first name"""
    try:
        doctor = Doctor.objects.get(first_name=doctor_name)
        return doctor
    except Doctor.DoesNotExist:
        return None


@ tool("get_all_doctor_detail")
def get_all_doctor_detail(
    doctor_name: Annotated[str, "Doctor's name"]
) -> List[Doctor]:
    """Tool to fetch information of a doctor from his first name"""
    try:
        doctors = Doctor.objects.filter()
        return doctors
    except Doctor.DoesNotExist:
        return None


@ tool("get_chat_history_from_patient_username")
def get_chat_history_from_patient_username(
    state: Annotated[dict, InjectedState]
) -> str:
    """Tool to fetch all the chat history of a patient with username"""
    # return state["messages"][-1].content
    # return "User chat history"
    try:
        user = User.objects.get(username=state["patient_username"])
        chats = Chat.objects.filter(user=user)
        return "Patient Chat History\n\n" + "\n".join(["Patient: "+x.message+"\nAnswer:"+x.response for x in chats])
    except User.DoesNotExist:
        return "No User History"


# Medication Node Tools

tavily_tool = TavilySearchResults(max_results=5)


@ tool("medical_stuff_search")
def medical_stuff_search(query: str) -> str:
    """Search with Google SERP API by a query to find information about medical stuff related to the query."""
    params = {
        "engine": "google",
        "gl": "us",
        "hl": "en",
    }
    # patent_search = SerpAPIWrapper(params=params, serpapi_api_key=serp_api_key)
    patent_search = SerpAPIWrapper(params=params)
    return patent_search.run(query)


@ tool("exa_search")
def exa_search(question: str) -> str:
    """Tool using Exa's Python SDK to run semantic search and return result highlights."""
    exa = Exa()

    response = exa.search_and_contents(
        question,
        type="neural",
        use_autoprompt=True,
        num_results=3,
        highlights=True
    )

    results = []
    for idx, eachResult in enumerate(response.results):
        result = {
            "Title": eachResult.title,
            "URL": eachResult.url,
            "Highlight": "".join(eachResult.highlights)
        }
        results.append(result)

    return json.dumps(results)


@ tool("knowledge_graph_medical_stuff_of_patient")
def knowledge_graph_medical_stuff_of_patient(
    state: Annotated[dict, InjectedState],
) -> Annotated[List[dict], "The information of patient's medical condition, medical regimen, diet, etc."]:
    """Tool to search for information about patient's medical condition, medical regimen, diet, etc. using the knowledge graph."""
    patientUsername = state["patient_username"]
    query = """
    MATCH (p:Patient)-[:REQUESTED]->(e:Entity)
    WHERE p.username = $patientUsername
    RETURN (e)
    """
    result = graphdb.run(query, patientUsername=patientUsername)
    nodeList = list(result)
    dictList = [dict(x[0]) for x in nodeList]
    return dictList


# Reservation Node Tools

@tool("get_reservation_detail_from_detailed_information")
def get_reservation_detail_from_detailed_information(
    state: Annotated[dict, InjectedState],
    doctor_name: Annotated[Optional[str], "Doctor's name"] = None,
    date: Annotated[Optional[date], "Reservation date"] = None,
    start_time: Annotated[Optional[time], "Reservation start time"] = None,
    end_time: Annotated[Optional[time], "Reservation end time"] = None
) -> str:
    """Tool to fetch all the reservations from doctor first name, reservation date, start time, end time"""
    try:
        user = User.objects.get(username=state["patient_username"])
        query = Q(user=user)

        if doctor_name:
            doctor = Doctor.objects.get(first_name=doctor_name)
            query &= Q(doctor=doctor)

        if date:
            query &= Q(date=date)

        if start_time:
            query &= Q(start_time=start_time)

        if end_time:
            query &= Q(end_time=end_time)

        reservations = Reservation.objects.filter(query)
        if len(reservations) == 1:
            x = reservations[0]
            return f""" Only one reservation matched! 
                Reservation:
                Doctor: {x.doctor.user.first_name} {x.doctor.user.last_name}
                Doctor's major: {x.doctor.major}
                Reservation date & time: {x.date} {x.start_time} ~ {x.end_time}
                Reason for reservation: {x.call_reason}
                """
        if len(reservations) > 0:
            return "Patient Reservation Detail\n\n" + "\n".join(
                [f"""
                Reservation:
                Doctor: {x.doctor.user.first_name} {x.doctor.user.last_name}
                Doctor's major: {x.doctor.major}
                Reservation date & time: {x.date} {x.start_time} ~ {x.end_time}
                Reason for reservation: {x.call_reason}
                """ for x in reservations])
        else:
            return "No reservaton matched with this information"

    except:
        return "No reservaton matched with this information"


@tool("get_reservation_detail_from_patient_username")
def get_reservation_detail_from_patient_username(
    state: Annotated[dict, InjectedState]
) -> str:
    """Tool to fetch all the reservation detail of a patient with username"""
    # return state["messages"][-1].content
    # return "User chat history"
    try:
        user = User.objects.get(username=state["patient_username"])
        patient = Patient.objects.get(user=user)
        reservations = Reservation.objects.filter(patient=patient)
        if len(reservations) > 0:
            return "Patient Reservation Detail\n\n" + "\n".join(
                [f"""
                Reservation:
                Doctor: {x.doctor.user.first_name} {x.doctor.user.last_name}
                Doctor's major: {x.doctor.major}
                Reservation date & time: {x.date} {x.start_time} ~ {x.end_time}
                Reason for reservation: {x.call_reason}
                """ for x in reservations])
        else:
            return "There is no reservation for this patient."
    except User.DoesNotExist:
        return "No User History"


@tool("get_reservation_detail_from_doctor_first_name")
def get_reservation_detail_from_doctor_name(
    doctor_name: Annotated[str, "Doctor's name"]
) -> List[str]:
    """Tool to fetch all the reservation detail of a doctor with first name"""
    # return state["messages"][-1].content
    # return "User chat history"
    try:
        doctor = Doctor.objects.get(first_name=doctor_name)
        reservations = Reservation.objects.filter(doctor=doctor)
        if len(reservations) > 0:
            return [f"""
                Reservation:
                Patient: {x.patient.user.first_name} {x.patient.user.last_name}
                Doctor's major: {x.doctor.major}
                Reservation date & time: {x.date} {x.start_time} ~ {x.end_time}
                Reason for reservation: {x.call_reason}
                """ for x in reservations]
        else:
            return ["No reservation for this doctor"]
    except User.DoesNotExist:
        return ["No reservation for this doctor"]


@tool("create_reservation_change_request")
def create_reservation_change_request(
    state: Annotated[dict, InjectedState],
    content: Annotated[str, "Content of reservation change request"]
) -> str:
    """Tool to save reservation change request to reservationRequest table"""
    tmp = state["patient_username"]
    user = User.objects.get(username=state["patient_username"])
    newRequest = ReservationRequest(
        user=user, content=content, created_at=timezone.now())
    newRequest.save()
    return f"Successfully created reservation change request, username: {tmp} content: {content}"


# Summarizer Node Tools


@tool("create_medical_stuff_node_in_knowledge_graph")
def create_medical_stuff_node_in_knowledge_graph(
    state: Annotated[Dict, InjectedState],
    extracted_entity: Annotated[dict, "Extracted entities from conversation"],
) -> Annotated[str, "Node ID"]:
    """
    Create a node of extract entities in the knowledge graph and make a connection with user node."""
    try:
        currentTime = str(timezone.now())
        extracted_entity["created_at"] = currentTime
        entityNode = Node("Entity", **extracted_entity)
        graphdb.create(entityNode)
    except:
        return "Not created the node"

    try:
        patientUsername = state["patient_username"]
        query = """MATCH (p:Patient), (e:Entity) 
            WHERE p.username = $patientUsername AND e.created_at = $currentTime
            CREATE (p)-[:REQUESTED]->(e) RETURN (p), (e)"""
        graphdb.run(query, patientUsername=patientUsername,
                    currentTime=currentTime)
        return "Successfully created!"
    except:
        deleteQuery = """MATCH (e:Entity)
        WHERE e.created_at = $currentTime
        DETACH DELETE e
        """
        graphdb.run(deleteQuery, currentTime=currentTime)
        return "Not created the relationship"

# Insight Node Tools


@tool("create_insight_record_in_postgres_db")
def create_insight_record_in_postgres_db(
    state: Annotated[Dict, InjectedState],
    content: Annotated[str, "Generated Medical Insight"],
) -> str:
    """Store generated insight into Insight table in PostgreSQL database"""
    try:
        delete_insight(state["patient_username"])
        create_insight(state["patient_username"], content)
        return "Successfully stored!"
    except:
        return "Storing failed!"


# Define Agents

def create_agent_for_classifier(llm, tools, system_message: str):
    """Create a agent that classifies the user input is related to medical and reservation-related things or not."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to determine if the user input is related to"
                " medical and reservation-related things or not."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If user input is related to medical things, "
                " prefix your response with MEDICATION CONTEXT so the team knows to continue to the medication node."
                " If user input is related to medical things, "
                " prefix your response with RESERVATION CONTEXT so the team knows to continue to the reservation node."
                " If user input requires insight of his medical condition, "
                " prefix your response with INSIGHT CONTEXT so the team knows to continue to the insight node."
                " If user input is not related to medical things nor reservation-related nor insight-related things, "
                " prefix your response with OUT OF CONTEXT so the team knows to stop right away."
                " \n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    return prompt | llm


def create_agent_for_nodes(llm, tools, system_message: str):
    """Create a agent that answers the user input that is related to medical or reservation-related things."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join(
        [tool.name for tool in tools]))

    return prompt | llm.bind_tools(tools)


def create_agent_for_summarizer(llm, tools, system_message: str):
    """Create a agent that extracts the name entities from conversation and then summarizes the conversation."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to extract name entities from conversation and store that into a graph database."
                " Execute what you can to make progress."
                " If you finish extracting the name entities from conversation and store that into a graph database,"
                " prefix you response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join(
        [tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


def create_agent_for_insight(llm, tools, system_message: str):
    """Create a agent that answers the user input that is related to medical or reservation-related things."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " Execute what you can to make progress."
                " If you have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join(
        [tool.name for tool in tools]))

    return prompt | llm.bind_tools(tools)

# Define Nodes


def agent_node(state, agent, name):
    result = agent.invoke(state)
    # print(state)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    if (state["sender"] == "Doctor" or state["sender"] == "Reservation_Assistant" or state["sender"] == "Insight"):
        messages = state["messages"]
        last_message = messages[-1]
        return {
            "messages": [result],
            "sender": name,
            "response": " ".join(last_message.content.split("FINAL ANSWER")),
        }
    elif state["sender"] == "Summarizer":
        return {
            "messages": [result],
            "sender": name,
            "response": state["response"],
        }

    return {
        "messages": [result],
        "sender": name,
    }

# Define Workflow

# Define router


def router_from_classfier(state) -> Literal["doctor", "reservation_assistant", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if "MEDICATION CONTEXT" in last_message.content:
        return "doctor"
    if "RESERVATION CONTEXT" in last_message.content:
        return "reservation_assistant"
    if "CONTINUE CONTEXT" in last_message.content:
        return "reservation_assistant"
    if "INSIGHT CONTEXT" in last_message.content:
        return "insight"
    if "OUT OF CONTEXT" in last_message.content:
        return "__end__"
    return "__end__"


def router_from_normal_nodes(state) -> Literal["call_tool", "continue", "__end__"]:
    messages = state["messages"]
    # print("#" * 10 + "          state")
    # print(state)
    last_message = messages[-1]
    # print(messages)
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "__end__"
    return "continue"


def router_from_summarizer(state) -> Literal["call_tool", "continue", "__end__"]:
    sender = state["sender"]
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "__end__"
    return "continue"


classifier_agent = create_agent_for_classifier(
    llm=llm,
    tools=[tavily_tool,
           get_patient_detail,
           get_chat_history_from_patient_username],
    system_message="""
        You should classfy the input into four categories:
        1. MEDICATION CONTEXT
        2. RESERVATION CONTEXT
        3. INSIGHT CONTEXT
        4. CONTINUE CONTEXT
        5. OUT OF CONTEXT
        Maybe patient will ask questions about himself or greetings. In that case, you classify that as MEDICATION CONTEXT.
        If patient asks insight for his medical condition or regimen or diet, you classify that into INSIGHT CONTEXT.
        Also if patient ask about the topic from the chat history, you classify that into suitable context so the team knows to continue. 
        For example, patient asked about how aspirin helps headache and is asking about what his former question was. In this case, 
        this question should be classified as CONTINUE CONTEXT.
        You should answer based on the chat history of a patient.
    """
)
classifier_node = functools.partial(
    agent_node,
    agent=classifier_agent,
    name="Classifier",
)

doctor_agent = create_agent_for_nodes(
    llm=llm,
    tools=[tavily_tool,
           get_chat_history_from_patient_username,
           get_patient_detail],
    system_message="""
        Patient will ask you about medical stuff or something about his medication or regimen or diet.
        Maybe he also ask you about something in the chat history.
        You should answer based on the chat history of a patient.
    """
)
doctor_node = functools.partial(
    agent_node,
    agent=doctor_agent,
    name="Doctor",
)

reservation_assistant_agent = create_agent_for_nodes(
    llm=llm,
    tools=[tavily_tool,
           get_patient_detail,
           get_chat_history_from_patient_username,
           get_doctor_detail,
           get_reservation_detail_from_patient_username,
           get_reservation_detail_from_doctor_name,
           get_reservation_detail_from_detailed_information,
           create_reservation_change_request],
    system_message="""
        You should answer the question based on information provided and chat history.
        1. You also need to check if the patient wants to create or reschedule or cancel a reservation.
        2. And in the case patient wants to reschedule or cancel the reservation, you must make sure which reservation patient wants to change.
        3. And in the case you patient wants to create the reservation, you must make sure the information patient provided is complete to create one.
        If yes, you should respond to the patient with something like 
        "I will convey your request to Dr. [Doctor's Name].
        Patient, [Patient Name] is requesting an appointment with Dr. [Doctor's Name] change from [Previous Detail] to [Requested Detail]."
        And add that to reservationRequest table.
        If not, you should ask for more information when you find it clear enough.
    """
)
reservation_assistant_node = functools.partial(
    agent_node,
    agent=reservation_assistant_agent,
    name="Reservation_Assistant",
)

summarizer_agent = create_agent_for_summarizer(
    llm=llm,
    tools=[create_medical_stuff_node_in_knowledge_graph],
    system_message="""
        You have to extract key entities from the conversation which the patient mentioned.
        For example, the patient's preference for appointment time, or any patient mention of a medication /diet /etc.
        If the patient says, "I am taking lisinopril twice a day" then extract {medication: lisinopril, frequency: twice a day}.
        This extracted entities and values should be used to create a node in the knowledge graph.
        If there are not any special key entities from the conversation you can skip the node creation.
        If you keep failing to create a node more than 5 times, you should finish.
    """
)
summarizer_node = functools.partial(
    agent_node,
    agent=summarizer_agent,
    name="Summarizer",
)

insight_agent = create_agent_for_insight(
    llm=llm,
    tools=[tavily_tool,
           get_all_doctor_detail,
           get_chat_history_from_patient_username,
           knowledge_graph_medical_stuff_of_patient,
           create_insight_record_in_postgres_db],
    system_message="""
        You have to give the insight of patient's medical condition, regimen, diet or treatment plan.
        First, you check the patient's detail.
        Second, you check the chat history and investigate patient's intention.
        Third, you have to give the step-by-step, detailed insight for patient and his treatment.
        Fourth, you can check the doctor's information and suggest to make a reservation with the suitable one.
        Fifth, you have to store generated insight into postgreSQL Database Insight table.
    """
)
insight_node = functools.partial(
    agent_node,
    agent=insight_agent,
    name="Insight",
)

tools = [
    tavily_tool,
    create_medical_stuff_node_in_knowledge_graph,
    get_chat_history_from_patient_username,
    get_patient_detail,
    get_doctor_detail,
    get_reservation_detail_from_patient_username,
    get_reservation_detail_from_doctor_name,
    get_reservation_detail_from_detailed_information,
    create_reservation_change_request,
    get_all_doctor_detail,
    knowledge_graph_medical_stuff_of_patient,
    create_insight_record_in_postgres_db
]
tool_node = ToolNode(tools)

workflow = StateGraph(HospitalAgentState)

workflow.add_node("Classifier", classifier_node)
workflow.add_node("Doctor", doctor_node)
workflow.add_node("Reservation_Assistant", reservation_assistant_node)
workflow.add_node("Summarizer", summarizer_node)
workflow.add_node("Insight", insight_node)
workflow.add_node("call_tool", tool_node)

workflow.add_edge(START, "Classifier")

workflow.add_conditional_edges(
    "Classifier",
    router_from_classfier,
    {"doctor": "Doctor", "reservation_assistant": "Reservation_Assistant",
        "insight": "Insight", "__end__": END}
)

workflow.add_conditional_edges(
    "Doctor",
    router_from_normal_nodes,
    {"continue": "Summarizer", "call_tool": "call_tool", "__end__": "Summarizer"}
)

workflow.add_conditional_edges(
    "Reservation_Assistant",
    router_from_normal_nodes,
    {"continue": "Summarizer", "call_tool": "call_tool", "__end__": "Summarizer"}
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {"Doctor": "Doctor", "Reservation_Assistant": "Reservation_Assistant",
        "Summarizer": "Summarizer", "Insight": "Insight"}
)

workflow.add_conditional_edges(
    "Summarizer",
    router_from_summarizer,
    {"continue": "Summarizer", "call_tool": "call_tool",
        "__end__": END}  # For Scalability
)

workflow.add_conditional_edges(
    "Insight",
    router_from_normal_nodes,
    {"continue": "Insight", "call_tool": "call_tool",
        "__end__": END}  # For Scalability
)

graph = workflow.compile()


def display_graph(graph):
    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass

# Utility functions for view functions


def user_exists(username):
    return User.objects.filter(username=username).exists()


# Create your views here.


def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')

        patient_username = "test"
        inputs = {
            "messages": [
                HumanMessage(
                    content=message
                )
            ],
            "sender": "human",
            "patient_username": patient_username,
            "response": ""
        }

        config = {"recursion_limit": 30, "configurable": {"thread_id": "2"}}

        events = graph.stream(
            inputs,
            config
        )

        for s in events:
            print(s)
            print('-' * 20)
        last_node = list(s.keys())[0]
        if last_node == "Classifier":
            response = "You are asking questions out of context. I will only answer the questions related to medication or reservation."
        elif last_node == "Insight":
            response = " ".join(s["Insight"]["messages"]
                                [0].content.split("FINAL ANSWER"))
        else:
            response = s[last_node]["response"]

        chat = Chat(user=request.user, message=message,
                    response=response, created_at=timezone.now())
        chat.save()
        return JsonResponse({'message': message, 'response': response})

    chats = Chat.objects.filter(user=request.user)
    reservationRequests = ReservationRequest.objects.filter(user=request.user)
    for chat in chats:
        chat.response = format_text(chat.response)
    return render(request, 'chatbot.html', {'chats': chats, 'reservations': reservationRequests})


def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')


def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        date_of_birth = dateparse.parse_datetime(request.POST['dob'])
        phone_number = request.POST['phone']
        location = request.POST['location']
        gender = request.POST['gender']

        if password1 != password2:
            error_message = 'Passwords do not match'
            return render(request, 'register.html', {'error_message': error_message})

        if user_exists(username):
            error_message = 'That username already exists!'
            return render(request, 'register.html', {'error_message': error_message})

        try:
            user = User.objects.create_user(username, email, password1)
            user.save()
            patient = Patient(
                user=user,
                first_name=first_name,
                last_name=last_name,
                phone_number=phone_number,
                location=location,
                date_of_birth=date_of_birth,
                gender=gender,
                created_at=timezone.now()
            )
            patient.save()
            auth.login(request, user)
            return redirect('chatbot')
        except:
            error_message = 'Error creating account'
            return render(request, 'register.html', {'error_message': error_message})

    return render(request, 'register.html')


def logout(request):
    auth.logout(request)
    return redirect('login')
