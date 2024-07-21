import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

from openai import AzureOpenAI
from streamlit_modal import Modal
from sentence_transformers import SentenceTransformer

from llm_agent import query_embeddings, get_prompt_template, get_memory_prompt_template

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource  # 👈 Add the caching decorator
def load_pubmed_model():
    return SentenceTransformer("neuml/pubmedbert-base-embeddings")

# Cache the CSV data
@st.cache_data
def load_csv(file_path, sep=','):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = pd.read_csv(f, sep=sep)
        return data
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

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
df_sponsors = load_csv('./data/adc_data.csv')
df_drugs = load_csv('./data/drugs.csv', sep='@')
df_trials = load_csv('./data/trials.csv')
df_trial_sponsors = load_csv('./data/trial_sponsors.csv')
df_indications = load_csv('./data/indications.csv')

model_pubmed = load_pubmed_model()
model = load_model()

index_name = 'adc'

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

with open('./logs/adc_logs.json', 'r', encoding='utf-8') as f:
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
    st.write("""This is a tracker for all ADC drugs under trials globally. 
             The data is updated daily and has been extracted from trials registries of United States, China, Japan, Europe, India and more than 10 other countries.""")
    
    # Select top 10 indications
    top_indications = df_indications[['Unique indications', 'Count']].head(10)

    st.subheader("Top Indications in Trials")
    fig_indications = px.bar(top_indications, x='Unique indications', y='Count', 
                             labels={'x': 'Indication', 'y': 'Count of Trials'}, title="Top Indications in Trials")
    fig_indications.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig_indications)
    
    # Top sponsors
    top_sponsors = df_trial_sponsors['sponsor'].value_counts().dropna().head(10)
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

    # Table view
    st.subheader("Top Indications & Sponsors")
    st.write(top_indications)
    st.write(top_sponsors)

# Individual sponsor view
def sponsor_view():
    st.title("Sponsor")
    # Selectbox to choose trial ID
    if st.session_state.selected_company:
        sponsor = st.selectbox("Select a Sponsor", df_sponsors['Company Name'].dropna().unique(), 
                            index=df_sponsors['Company Name'].dropna().unique().tolist().index(st.session_state.selected_company))
    else:
        sponsor = st.selectbox("Select a Sponsor", df_sponsors['Company Name'].dropna().unique())
    
    # Update session state with the selected trial ID
    if sponsor != st.session_state.selected_company:
        st.session_state.selected_company = sponsor
    
    sponsor_data = df_sponsors[df_sponsors['Company Name'] == sponsor].reset_index()
    filtered_drugs_df = df_drugs[df_drugs['developers'].apply(lambda x: pd.notna(x) and (sponsor in [dev.strip() for dev in x.split('|')]))]
    filtered_trials_df = df_trial_sponsors[df_trial_sponsors['sponsor'] == sponsor].dropna().drop_duplicates().reset_index()
    
    if len(sponsor_data):
        details = sponsor_data.iloc[0].to_dict()
        st.subheader(f"Overview of {sponsor}")
        st.write(details)

    st.subheader(f"List of Drugs for {sponsor}")
    
    st.write(filtered_drugs_df.reset_index()[['heading', 'domain', 'Max Phase', 'developers']])

    # Clickable trials for sponsor
    st.write("Click on a trial ID to see details for that trial.")
    trial_id = st.selectbox("Select a Trial ID", filtered_trials_df['id'].unique(), key='sponsor_trials')
    if trial_id:
        if st.button("Go to Trial"):
            navigate_to("Trial", trial=trial_id)
    
    st.subheader(f"List of Trials for {sponsor}")
    st.write(filtered_trials_df[['id']])

# Individual trial view
def trial_view():
    st.title("Trial")
    # Selectbox to choose trial ID
    if st.session_state.selected_trial:
        trial_id = st.selectbox("Select a Trial ID", df_trials['id'].dropna().unique(), 
                            index=df_trials['id'].dropna().unique().tolist().index(st.session_state.selected_trial))
    else:
        trial_id = st.selectbox("Select a Trial ID", df_trials['id'].dropna().unique())
    
    # Update session state with the selected trial ID
    if trial_id != st.session_state.selected_trial:
        st.session_state.selected_trial = trial_id
    
    trial_data = df_trials[df_trials['id'] == trial_id].iloc[0]
    
    st.subheader(f"Details for Trial ID: {trial_id}")
    if pd.notna(trial_data['sponsor']):
        st.write(f"**Sponsor:** {trial_data['sponsor']}")
    if pd.notna(trial_data['collaborators']):
        st.write(f"**Collaborators:** {trial_data['collaborators']}")
    
    if pd.notna(trial_data['conditions']):
        st.write(f"**Conditions:** {trial_data['conditions']}")
    
    if pd.notna(trial_data['interventions']):
        st.write(f"**Intervention:** {trial_data['interventions']}")
    
    if pd.notna(trial_data['phases']):
        st.write(f"**Phase:** {trial_data['phases']}")


sponsor_prompt = """Given the user's message, provide a 1-3 word phrase to search for the most relevant sponsors in the database.
                    Response with just "None" if the message doesn't mention anything about a sponsor name."""


def handle_prompt():
    prompt = st.chat_input("What are you looking for?")
    if prompt:
        if len(prompt) > 2048:
            with st.chat_message("assistant"):
                response = st.write_stream("Message is too large.")
            st.session_state.messages.append({"role": "assistant", "content": response})
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.logs.append(prompt)
        with open('./logs/adc_logs.json', 'w') as f:
            data = {"logs": st.session_state.logs}
            json.dump(data, f, ensure_ascii=False, indent=4)

        with st.chat_message("user"):
            st.markdown(prompt)

        vector = model.encode(prompt)
        response = query_embeddings(vector, n_results=5, embedding_name=index_name, pc_api_key=pc_api_key)
        context_docs = [v['metadata'] for v in response]

        print(f"Context Docs: {context_docs}\n\n")
        
        template = get_prompt_template(prompt, context_docs)
        
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
    st.title("💬 ADC Tracker Chat")
    st.write(
        """Interact with this chatbot to answer your questions on ADC drug development across the world."""
    )

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    handle_prompt()


def run():
    st.sidebar.image('./utils/lunartree-logo-256256.png', width=200)  # Adjust width as needed
    if st.sidebar.button("Home"):
        st.session_state.page = None
        st.rerun()

    st.sidebar.title("LunarTree ADC Tracker")

    view = st.sidebar.radio("Select a view", ["Summary", "Sponsor", "Trial"], 
                            index=["Summary", "Sponsor", "Trial"].index(st.session_state.view))

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
        trial_view()
        # chat_view()

    # Modal component
    modal = Modal("Welcome to the ADC Tracker!", key="modal")

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