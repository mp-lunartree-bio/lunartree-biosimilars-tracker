import streamlit as st
# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = None

if st.session_state.page is None:
    st.title("Welcome to the LunarTree Tracker")
    st.subheader("Choose the drug category you are interested in:")

    if st.button("Biosimilar", key='biosimilar'):
        st.session_state.page = 'Biosimilar'
        st.rerun()

    if st.button("Antibody Drug Conjugates", key='adc'):
        st.session_state.page = 'ADC'
        st.rerun()
else:
    if st.session_state.page == 'Biosimilar':
        import biosimilar
        biosimilar.run()
    elif st.session_state.page == 'ADC':
        import adc
        adc.run()