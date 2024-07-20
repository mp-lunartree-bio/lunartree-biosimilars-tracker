import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

from openai import AzureOpenAI
from streamlit_modal import Modal
from sentence_transformers import SentenceTransformer

from llm_agent import query_embeddings, get_prompt_template, get_memory_prompt_template

# Set page configuration to wide mode
st.set_page_config(layout="wide")

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource  # 👈 Add the caching decorator
def load_pubmed_model():
    return SentenceTransformer("neuml/pubmedbert-base-embeddings")

# Load JSON data
@st.cache_data
def load_data():
    with open('biosimilar_trial_data.json', 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

@st.cache_data
def load_sponsors():
    with open('sponsors.csv', 'r') as f:
        data = pd.read_csv(f)
    return data

openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"]
openai_ep = st.secrets["AZURE_OPENAI_ENDPOINT"]
openai_ver = "2024-02-01"
pc_api_key = st.secrets['PINECONE_API_KEY']

os.environ['AZURE_OPENAI_ENDPOINT'] = st.secrets["AZURE_OPENAI_ENDPOINT"]
os.environ["OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
os.environ['SERPER_API_KEY'] = st.secrets["SERPAPI_API_KEY"]
os.environ['OPENAI_API_VERSION'] = openai_ver

if not openai_api_key:
    raise ValueError("Please enter your OpenAI API key.")
else:
    client = AzureOpenAI(
            azure_endpoint=openai_ep,
            api_key=openai_api_key,
            api_version=openai_ver,
        )

df = load_data()
df_sponsors = load_sponsors()
model_pubmed = load_pubmed_model()
model = load_model()

index_name = 'trials'
index_name_pubmed = 'trial-pubmed'
csv_index_name = 'sponsors'

# Dictionary to store mappings of lowercase values to canonical representations
canonical_values = {}

# Iterate over each column
for col in ['Disease Name', 'Novel Drug Name', 'Reference Drug Generic Name']:
    # Convert column values to lowercase and iterate row by row
    for index, value in df[col].items():
        # Convert current value to lowercase
        if isinstance(value, str):
            lower_value = value.lower()
            
            # Check if lower_value has been encountered before
            if lower_value in canonical_values:
                # Replace current value with canonical value
                df.at[index, col] = canonical_values[lower_value]
            else:
                # Store canonical mapping for future references
                canonical_values[lower_value] = value

# Initialize session state
if 'view' not in st.session_state:
    st.session_state.view = "Summary"
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = None
if 'selected_trial' not in st.session_state:
    st.session_state.selected_trial = None
if 'modal_opened' not in st.session_state:
    st.session_state.modal_opened = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'logs' not in st.session_state:
    st.session_state.logs = []

with open('./logs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    if 'logs' not in data:
        data = {'logs': []}
    st.session_state.logs = data['logs']


# Navigation function
def navigate_to(view, company=None, trial=None):
    if company:
        st.session_state.selected_company = company
    if trial:
        st.session_state.selected_trial = trial
    
    st.session_state.view = view
    st.rerun()

# Summary level view
def aggregate_view():
    st.title("Tracker Summary")
    st.write("""This is a tracker for all biosimilar drugs under trials globally. 
             The data is updated daily and has been extracted from trials registries of United States, China, Japan, Europe, India and more than 10 other countries.""")
    
    # Top indications
    top_indications = df['Disease Name'].value_counts().dropna().head(10)
    top_indications.columns = ['Count of Trials']

    st.subheader("Top Indications in Trials")
    fig_indications = px.bar(top_indications, x=top_indications.index, y=top_indications.values, 
                             labels={'x': 'Indication', 'y': 'Count of Trials'}, title="Top Indications in Trials")
    fig_indications.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig_indications)
    
    # Top sponsors
    top_sponsors = df['Sponsor'].value_counts().dropna().head(10)
    top_sponsors.columns = ['Count of Trials']
        
    # Sponsor dropdown for navigation
    sponsor = st.selectbox("Select a Sponsor to view details", top_sponsors.index)
    if sponsor:
        if st.button("Go to Sponsor"):
            navigate_to("Sponsor", company=sponsor)

    st.subheader("Top Sponsors with Number of Trials")
    fig_sponsors = px.bar(top_sponsors, x=top_sponsors.index, y=top_sponsors.values, 
                             labels={'x': 'Sponsor', 'y': 'Count of Trials'}, title="Top Sponsors with Number of Trials")
    fig_sponsors.update_layout(xaxis={'categoryorder':'total descending'})
    fig_sponsors.update_traces(marker_color='indianred')
    fig_sponsors.update_traces(marker=dict(line=dict(color='#000000', width=2)))
    fig_sponsors.update_traces(hovertemplate='%{x}<extra></extra>')
    st.plotly_chart(fig_sponsors)
    
    # Top reference drugs
    top_reference_drugs = df['Reference Drug Generic Name'].value_counts().dropna().head(10)
    top_reference_drugs.columns = ['Count of Trials']

    st.subheader("Top Reference Drugs")
    fig_references = px.bar(top_reference_drugs, x=top_reference_drugs.index, y=top_reference_drugs.values, 
                             labels={'x': 'Reference Drug', 'y': 'Count of Trials'}, title="Top Reference Drugs")
    fig_references.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig_references)

    # Table view
    st.subheader("Top Indications, Sponsors, and Reference Drugs")
    st.write(top_indications)
    st.write(top_sponsors)
    st.write(top_reference_drugs)

# Individual sponsor view
def sponsor_view():
    st.title("Sponsor")
    # Selectbox to choose trial ID
    if st.session_state.selected_company:
        sponsor = st.selectbox("Select a Sponsor", df['Sponsor'].dropna().unique(), 
                            index=df['Sponsor'].dropna().unique().tolist().index(st.session_state.selected_company))
    else:
        sponsor = st.selectbox("Select a Sponsor", df['Sponsor'].dropna().unique())
    
    # Update session state with the selected trial ID
    if sponsor != st.session_state.selected_company:
        st.session_state.selected_company = sponsor
    
    sponsor_data = df[df['Sponsor'] == sponsor].reset_index()
    sponsor_details = df_sponsors[df_sponsors['Sponsor Name'] == sponsor]
    
    if len(sponsor_details):
        details = sponsor_details.iloc[0].to_dict()
        st.subheader(f"Overview of {sponsor}")
        st.write(f"**Location:** {details['Location']}")
        st.write(f"**Brief Overview:** {details['Overview']}")

    st.subheader(f"List of Drugs for {sponsor}")
    st.write(sponsor_data[['Novel Drug Name','Reference Drug Generic Name']].dropna().drop_duplicates().reset_index())

    # Clickable trials for sponsor
    st.write("Click on a trial ID to see details for that trial.")
    trial_id = st.selectbox("Select a Trial ID", sponsor_data['TrialID'].unique(), key='sponsor_trials')
    if trial_id:
        if st.button("Go to Trial"):
            navigate_to("Trial", trial=trial_id)
    
    st.subheader(f"List of Trials for {sponsor}")
    st.write(sponsor_data[['TrialID', 'Disease Name', 'Novel Drug Name', 'Reference Drug Generic Name']])

    st.subheader(f"Top Indications Targeted by {sponsor}")
    st.write(sponsor_data['Disease Name'].value_counts().dropna().head(10))

# Individual trial view
def trial_view():
    st.title("Trial")
    # Selectbox to choose trial ID
    if st.session_state.selected_trial:
        trial_id = st.selectbox("Select a Trial ID", df['TrialID'].dropna().unique(), 
                            index=df['TrialID'].dropna().unique().tolist().index(st.session_state.selected_trial))
    else:
        trial_id = st.selectbox("Select a Trial ID", df['TrialID'].dropna().unique())
    
    # Update session state with the selected trial ID
    if trial_id != st.session_state.selected_trial:
        st.session_state.selected_trial = trial_id
    
    trial_data = df[df['TrialID'] == trial_id].iloc[0]
    
    st.subheader(f"Details for Trial ID: {trial_id}")
    if pd.notna(trial_data['Sponsor']):
        st.write(f"**Sponsor:** {trial_data['Sponsor']}")
    
    if pd.notna(trial_data['Disease Name']):
        st.write(f"**Indication:** {trial_data['Disease Name']}")
    elif pd.notna(trial_data['Condition']):
        st.write(f"**Indication:** {trial_data['Condition']}")
    
    if pd.notna(trial_data['Novel Drug Name']):
        st.write(f"**Drug:** {trial_data['Novel Drug Name']}")
    elif pd.notna(trial_data['Intervention']):
        st.write(f"**Drug:** {trial_data['Intervention']}")
    
    if pd.notna(trial_data['Reference Drug Generic Name']):
        st.write(f"**Reference Drug:** {trial_data['Reference Drug Generic Name']}")
    
    if pd.notna(trial_data['Phase']):
        st.write(f"**Phase:** {trial_data['Phase']}")
    
    other_details = trial_data.dropna().to_dict()
    if other_details:
        st.write("**Other Details:**")
        st.json(other_details)


sponsor_prompt = """Given the user's message, provide a 1-3 word phrase to search for the most relevant sponsors in the database.
                    Response with just "None" if the message doesn't mention anything about a sponsor name."""


def handle_prompt():
    prompt = st.chat_input("What are you looking for?")
    if prompt:
        if len(prompt) > 2048:
            with st.chat_message("assistant"):
                response = st.write_stream(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.logs.append(prompt)
        with open('./logs.json', 'w') as f:
            data = {"logs": st.session_state.logs}
            json.dump(data, f, ensure_ascii=False, indent=4)

        with st.chat_message("user"):
            st.markdown(prompt)

        response = client.chat.completions.create(
            model='lunartree-gpt-35-turbo-2',
            messages=[{"role": "system", "content": sponsor_prompt}, {"role": "user", "content": prompt}],
        )

        sponsor_query = response.choices[0].message.content
        print(f"Sponsor Query: {sponsor_query}\n\n")
        matching_query = ""
        if "None" not in sponsor_query:
            vector = model.encode(sponsor_query)
            response = query_embeddings(vector, n_results=2, embedding_name=csv_index_name, pc_api_key=pc_api_key)
            sponsor_match = response[0]

            if sponsor_match['score'] > 0.4:
                metadata = sponsor_match['metadata']
                matching_query = f" Matching Sponsor: {metadata['Sponsor Name']}, {metadata.get('Other Names')}. "

            print([(v['id'], v['score']) for v in response])

        memory_prompt = get_memory_prompt_template(st.session_state.messages[-5:])

        if len(memory_prompt):
            response = client.chat.completions.create(
                model='lunartree-gpt-35-turbo-2',
                messages=[{"role": "user", "content": memory_prompt}],
            )

            query = response.choices[0].message.content
            print(f"Refined Query: {query}\n\n")
        else:
            query = prompt

        context_trials = []

        trial_query = f"{query}. {matching_query}"
        print(f"Trial Query {trial_query}\n\n")
        vector = model.encode(trial_query)
        response = query_embeddings(vector, n_results=5, embedding_name=index_name, pc_api_key=pc_api_key)
        context_trials.extend([v['metadata'] for v in response])

        print([(v['id'], v['score'], v['metadata']) for v in response])

        vector = model_pubmed.encode(trial_query)
        response = query_embeddings(vector, n_results=5, embedding_name=index_name_pubmed, pc_api_key=pc_api_key)
        context_trials.extend([v['metadata'] for v in response])

        print([(v['id'], v['score'], v['metadata']) for v in response])

        trial_ids = []
        unique_trials = []
        for trial in context_trials:
            if trial['TrialID'] not in trial_ids:
                unique_trials.append(trial)
                trial_ids.append(trial['TrialID'])

        print(f"Unique Trials: {unique_trials}\n\n")
        
        template = get_prompt_template(query, unique_trials)
        
        response = client.chat.completions.create(
            model='lunartree-gpt-35-turbo-2',
            messages=[{"role": "user", "content": template}],
            stream=True
        )

        with st.chat_message("assistant"):
            response = st.write_stream(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def chat_view():
    # Show title and description.
    st.title("💬 Biosimilar Tracker Chat")
    st.write(
        """Interact with this chatbot to answer your questions on biosimilar drug development across the world."""
    )

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    handle_prompt()

st.sidebar.image('lunartree-logo-256256.png', width=200)  # Adjust width as needed
st.sidebar.title("LunarTree Biosimilars Tracker")

view = st.sidebar.radio("Select a view", ["Summary", "Sponsor", "Trial", "Chat"], 
                        index=["Summary", "Sponsor", "Trial", "Chat"].index(st.session_state.view))

# Buttons for website, LinkedIn, and email
st.sidebar.markdown("## Connect with Us")
st.sidebar.markdown("""
<a href="https://www.lunartree.bio" target="_blank"><button style="background-color: #000000; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">Visit our Website</button></a>
<a href="https://www.linkedin.com/in/mukeshpareek1997/" target="_blank"><button style="background-color: #000000; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">Connect on LinkedIn</button></a>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Email Us")
st.sidebar.markdown("contact@lunartree.bio")

# Update the session state view if the radio button changes
if view != st.session_state.view:
    st.session_state.view = view
    st.rerun()

# Display the correct view based on session state
if st.session_state.view == "Summary":
    aggregate_view()
elif st.session_state.view == "Sponsor":
    sponsor_view()
elif st.session_state.view == "Trial":
    trial_view()
else:
    chat_view()

# Modal component
modal = Modal("Welcome to the Biosimilars Tracker!", key="modal")

if not st.session_state.modal_opened:
    st.session_state.modal_opened = True
    modal.open()

if modal.is_open():
    with modal.container():
        st.markdown("""
        LunarTree AI agents automatically search and extract data from clinical trials and company websites to generate the data powering this dashboard. 
        
        <span style="color: green; font-weight: bold;">This is a sample dashboard generated by our tool within 3 hours.</span>
        
        It can be configured to update daily with fresh data.

        These dashboards can be customized for specific drug categories, companies, or indications at a fraction of the time and cost typically required by consultants or in-house teams. 

        We are adding more exciting capabilities and developing dashboards for various data sources and therapy areas. 
        
        <span style="color: green; font-weight: bold;">Get in touch to discuss your custom use cases.</span>

        """, unsafe_allow_html=True)
        if st.button("Proceed"):
            modal.close()