import requests
import csv
from datetime import datetime, timedelta
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, StuffDocumentsChain
from langchain_openai import ChatOpenAI
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
import tiktoken
import hashlib

spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
)
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
security = "https://seit.egr.msu.edu/data/security_title_url.csv"
iot = "https://seit.egr.msu.edu/data/ai_title_url.csv"
blockchain = "https://seit.egr.msu.edu/data/blockchain_title_url.csv"
ai = "https://seit.egr.msu.edu/data/iot_title_url.csv"

link_dict = {
    "security": "https://seit.egr.msu.edu/data/security_title_url.csv",
    "iot": "https://seit.egr.msu.edu/data/ai_title_url.csv",
    "blockchain": "https://seit.egr.msu.edu/data/blockchain_title_url.csv",
    "ai": "https://seit.egr.msu.edu/data/iot_title_url.csv"
}
llm = ChatOpenAI()
stuff_link_summarize = load_summarize_chain(llm, chain_type='stuff', verbose=True)
map_reduce_link_summarize = load_summarize_chain(llm, chain_type='map_reduce', verbose=True)
stuff_topic_summarize = load_summarize_chain(llm, chain_type='stuff', verbose=True)

map_template = """The following is a set of documents
{docs}
Based on this list of docs, Identify the main themes and concisely summarize the documents in an article like tone and format. \
Format your summary in markdown and make sure to include the [one alphanumeric source] notation whenever you use information from the article summaries provided to you. \
Place these citations at the end of the sentence or paragraph that reference them - do NOT put them all at the end. \
Do NOT include multiple sources within one pair of brackets. There should only be one source per pair of brackets. \
Summary:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=True)

reduce_template = """You are a journalist who is an expert at summarizing technical articles. \
Here are a list of article summaries: \
{docs}
Write a summary of the article summaries in an article like tone and format. \
Cite your sources using [one alphanumeric source] notation which maps to the source of the article summaries provided to you. \
Be fairly in depth when needed, but also to the point. \
Format your summary in markdown and make sure to include the [one alphanumeric source] notation whenever you use information from the article summaries provided to you.
Place these citations at the end of the sentence or paragraph that reference them - do NOT put them all at the end. \
Do NOT include multiple sources within one pair of brackets. There should only be one source per pair of brackets. \
Your markdown formatted response should be easy to read, concise and information dense. \
Remember: Format your summary in markdown and make sure to include the [one alphanumeric source] notation whenever you use information from the article summaries provided to you. \
Article/Summary:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=True)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs", verbose=True
)
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=14000,
    verbose=True
)
map_reduce_topic_summarize = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
    verbose=True
)

def fetch_links_from_past_hours(link_dict=link_dict, hours=24):
    # Fetch the data from the URL
    response_dict = dict()
    for key, value in link_dict.items():
        response = requests.get(value)
        if response.status_code == 200:
            content = response.content.decode('utf-8')
            csv_data = list(csv.reader(content.splitlines()))
            current_time = datetime.now()
            time_hours_ago = current_time - timedelta(hours=hours)
            filtered_links = []
            for row in csv_data[1:]:
                if len(row) >= 4:
                    timestamp_str = row[3]
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    if timestamp >= time_hours_ago:
                        filtered_links.append(row[2])
            response_dict[key] = filtered_links
        else:
            print(f"Failed to fetch the data from the URL: {value}")
            response_dict[key] = []
    return response_dict


def extract_from_links(links_dict):
    extracted_data = dict()
    for key, value in links_dict.items():
        data = []
        for link in value:
            loader = UnstructuredURLLoader([link])
            docs = loader.load()
            if docs:
                docs[0].metadata['link'] = link
                data.append(docs[0])
        extracted_data[key] = data
    return extracted_data


def summarize_link(docs):
    if len(tokenizer.encode(docs[0].page_content)) > 14000:
        summary = map_reduce_link_summarize.run(docs)
        return Document(page_content=summary, metadata=docs[0].metadata)
    summary = stuff_link_summarize.run(docs)
    return Document(page_content=summary, metadata=docs[0].metadata)


def summarize_topic(docs: list[Document]):
    for doc in docs:
        doc.metadata.update({'source': hash_string(doc.metadata['link'])})
        doc.page_content = doc.page_content + f"[{doc.metadata['source']}]"
    topic_summary_1 = map_reduce_topic_summarize.run(docs)
    topic_summary = topic_summary_1
    count = 1
    link_source_map = dict()
    for doc in docs:
        if f"[{doc.metadata['source']}]" in topic_summary:
            topic_summary = topic_summary.replace(f"[{doc.metadata['source']}]", f"[[{count}]]({doc.metadata['link']})")
            link_source_map[f"[{doc.metadata['source']}]"] = doc.metadata['link']
            count += 1
    return topic_summary, topic_summary_1, link_source_map

def final_summary(summaries: list[str], link_source_maps: list[dict[str, str]]):
    docs = [Document(page_content=summary) for summary in summaries]
    summary = map_reduce_topic_summarize.run(docs)
    link_source_maps = {
        key: value for link_source_map in link_source_maps for key, value in link_source_map.items()
    }
    count = 1
    print(f"Summaries: {summaries}")
    print(f"Link Source Maps: {link_source_maps}")
    print(f"Summary: {summary}")
    for key, value in link_source_maps.items():
        print("Replacing:", key)
        summary = summary.replace(f"{key}", f"[[{count}]]({value})")
        count += 1
    return summary


def hash_string(string):
    hashed = hashlib.sha256(string.encode()).hexdigest()
    return hashed[:7]