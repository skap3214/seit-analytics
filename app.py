import streamlit as st
from fetch import fetch_links_from_past_hours, extract_from_links, summarize_link, summarize_topic, final_summary


st.title("Latest SEIT Posts Summary")

if generate := st.button('Generate Summary'):
    # security, ai, iot, blockchain = st.tabs(['Security', 'AI', 'IoT', 'Blockchain'])
    # security, ai, iot, blockchain = st.columns(4)
    with st.spinner(f'Fetching links'):
        links = fetch_links_from_past_hours()
        docs_dict = extract_from_links(fetch_links_from_past_hours())

    def topic_summary_generate(topic: str):
        with st.spinner(f'Analyzing articles'):
            topic_docs = docs_dict[topic]
            if topic_docs == []:
                st.write("No articles published in the last 24 hours.")
                return None, dict()
            summaries = []
            for doc in topic_docs:
                summary = summarize_link([doc])
                summaries.append(summary)
        with st.spinner(f'Summarizing topic'):
            topic_summary, topic_summary_1, link_source_map = summarize_topic(summaries)
            st.markdown(topic_summary)
            print("Topic Summary 1:", topic_summary_1)
            return topic_summary_1, link_source_map
        
    with st.expander('Security', expanded=True) as security:
        st.subheader('Security')
        security_summary, security_links = topic_summary_generate('security')
    with st.expander('AI', expanded=True) as ai:
        st.subheader('AI')
        ai_summary, ai_links = topic_summary_generate('ai')
    with st.expander('IoT', expanded=True) as iot:
        st.subheader('IoT')
        iot_summary, iot_links = topic_summary_generate('iot')
    with st.expander('Blockchain', expanded=True) as blockchain:
        st.subheader('Blockchain')
        blockchain_summary, blockchain_links = topic_summary_generate('blockchain')
    all_summary = [security_summary, ai_summary, iot_summary, blockchain_summary]
    all_links = [security_links, ai_links, iot_links, blockchain_links]
    fs = final_summary([summary for summary in all_summary if summary != None], all_links)
    st.markdown(fs)