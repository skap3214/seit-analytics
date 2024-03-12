import streamlit as st
from fetch import fetch_links_from_past_hours, extract_from_links, summarize_link, summarize_topic

@st.cache_data
def extract():
    return extract_from_links(fetch_links_from_past_hours())

st.title('SEIT Paper Summaries')

if generate := st.button('Generate Summary'):
    # security, ai, iot, blockchain = st.tabs(['Security', 'AI', 'IoT', 'Blockchain'])
    # security, ai, iot, blockchain = st.columns(4)
    with st.spinner(f'Fetching links'):
        links = fetch_links_from_past_hours()
        docs_dict = extract_from_links(links)

    @st.cache_data
    def topic_summary_generate(topic: str):
        with st.spinner(f'Analyzing articles'):
            topic_docs = docs_dict[topic]
            if topic_docs == []:
                st.write("No articles published in the last 24 hours.")
                return
            summaries = []
            for doc in topic_docs:
                summary = summarize_link([doc])
                summaries.append(summary)
        with st.spinner(f'Summarizing topic'):
            topic_summary, topic_summary_1 = summarize_topic(summaries)
            st.markdown(topic_summary)

    with st.expander('Security', expanded=True) as security:
        st.subheader('Security')
        topic_summary_generate('security')
    with st.expander('AI', expanded=True) as ai:
        st.subheader('AI')
        topic_summary_generate('ai')
    with st.expander('IoT', expanded=True) as iot:
        st.subheader('IoT')
        topic_summary_generate('iot')
    with st.expander('Blockchain', expanded=True) as blockchain:
        st.subheader('Blockchain')
        topic_summary_generate('blockchain')
    
    blockchain.expanded = False