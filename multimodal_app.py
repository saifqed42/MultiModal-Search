import streamlit as st
import os
import soundfile as sf
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import torchaudio
import torch
from transformers import ClapModel, ClapProcessor
from typing import Optional, Sequence, List, Dict, Union
from chromadb.api.types import Document, Embedding, EmbeddingFunction, URI, DataLoader
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from PIL import Image
import cv2

path = "mm_vdb"
audio_folder = "esc50"
image_folder = "StockImages-cc0"
video_folder = "StockVideos-CC0"
video_frames_folder = "StockVideos-CC0-frames"
wiki_embeddings_file = "wikipedia_embeddings.json"

st.title("Multimodal Search App (Audio, Image, Text, and Video)")

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=path)

client = get_chroma_client()

class AudioLoader(DataLoader[List[Optional[Dict[str, any]]]]):
    def __init__(self, target_sample_rate: int = 48000) -> None:
        self.target_sample_rate = target_sample_rate

    def _load_audio(self, uri: Optional[URI]) -> Optional[Dict[str, any]]:
        if uri is None:
            return None
        try:
            waveform, sample_rate = torchaudio.load(uri)
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            return {"waveform": waveform.squeeze(), "uri": uri}
        except Exception as e:
            st.error(f"Error loading audio file {uri}: {str(e)}")
            return None

    def __call__(self, uris: Sequence[Optional[URI]]) -> List[Optional[Dict[str, any]]]:
        return [self._load_audio(uri) for uri in uris]

class CLAPEmbeddingFunction(EmbeddingFunction[Union[Document, Dict[str, any]]]):
    def __init__(
        self,
        model_name: str = "laion/larger_clap_general",
        device: str = None
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.device = device

    def _encode_audio(self, audio: torch.Tensor) -> Embedding:
        inputs = self.processor(audios=audio.numpy(), sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            audio_embedding = self.model.get_audio_features(**inputs)
        return audio_embedding.squeeze().cpu().numpy().tolist()

    def _encode_text(self, text: Document) -> Embedding:
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)
        return text_embedding.squeeze().cpu().numpy().tolist()

    def __call__(self, input: Union[List[Document], List[Optional[Dict[str, any]]]]) -> List[Optional[Embedding]]:
        embeddings = []
        for item in input:
            if isinstance(item, dict) and 'waveform' in item:
                embeddings.append(self._encode_audio(item['waveform']))
            elif isinstance(item, str):
                embeddings.append(self._encode_text(item))
            elif item is None:
                embeddings.append(None)
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return embeddings

@st.cache_resource
def get_audio_collection():
    return client.get_or_create_collection(
        name='audio_collection',
        embedding_function=CLAPEmbeddingFunction(),
        data_loader=AudioLoader()
    )

audio_collection = get_audio_collection()

@st.cache_resource
def get_image_collection():
    return client.get_or_create_collection(
        name='image_collection',
        embedding_function=OpenCLIPEmbeddingFunction(),
        data_loader=ImageLoader()
    )

image_collection = get_image_collection()

@st.cache_resource
def get_text_collection():
    return client.get_or_create_collection(name="text_collection")

text_collection = get_text_collection()

@st.cache_resource
def get_video_collection():
    return client.get_or_create_collection(
        name='video_collection',
        embedding_function=OpenCLIPEmbeddingFunction(),
        data_loader=ImageLoader()
    )

video_collection = get_video_collection()

def add_audio(audio_collection, folder_path):
    ids = []
    uris = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_id = os.path.splitext(filename)[0]
            file_uri = os.path.join(folder_path, filename)
            ids.append(file_id)
            uris.append(file_uri)
    audio_collection.add(ids=ids, uris=uris)

def add_images(image_collection, folder_path):
    ids = []
    uris = []
    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith('.jpg'):
            file_uri = os.path.join(folder_path, filename)
            ids.append(str(i))
            uris.append(file_uri)
    image_collection.add(ids=ids, uris=uris)

@st.cache_resource
def generate_wiki_embeddings():
    ds = load_dataset("TopicNavi/Wikipedia-example-data", split="train")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    
    documents = ds['text']
    metadatas = [{"url": entry, "wiki_id": id} for entry, id in zip(ds['url'], ds['wiki_id'])]
    ids = ds['title']
    
    embeddings = model.encode(documents, convert_to_tensor=True, device=device)
    embeddings = embeddings.cpu().numpy().tolist()
    
    return documents, embeddings, metadatas, ids

def load_or_generate_wiki(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['documents'], data['embeddings'], data['metadatas'], data['ids']
    else:
        st.warning("Wikipedia embeddings file not found. Generating embeddings...")
        docs, embs, metas, ids = generate_wiki_embeddings()
        
        data = {
            "documents": docs,
            "embeddings": embs,
            "metadatas": metas,
            "ids": ids
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        st.success("Wikipedia embeddings generated and saved.")
        return docs, embs, metas, ids

def batch_add_to_collection(collection, documents, embeddings, metadatas, ids, batch_size=5461):
    for i in range(0, len(documents), batch_size):
        doc_batch = documents[i:i + batch_size]
        emb_batch = embeddings[i:i + batch_size]
        meta_batch = metadatas[i:i + batch_size]
        id_batch = ids[i:i + batch_size]
        collection.add(
            documents=doc_batch,
            embeddings=emb_batch,
            metadatas=meta_batch,
            ids=id_batch
        )
        print(f"Batch {i // batch_size + 1} added to the text collection successfully.")

def extract_frames(video_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_filename in os.listdir(video_folder):
        if video_filename.endswith('.mp4'):
            video_path = os.path.join(video_folder, video_filename)
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps

            output_subfolder = os.path.join(output_folder, os.path.splitext(video_filename)[0])
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            success, image = video_capture.read()
            frame_number = 0
            while success:
                if frame_number == 0 or frame_number % int(fps * 5) == 0 or frame_number == frame_count - 1:
                    frame_time = frame_number / fps
                    output_frame_filename = os.path.join(output_subfolder, f'frame_{int(frame_time)}.jpg')
                    cv2.imwrite(output_frame_filename, image)

                success, image = video_capture.read()
                frame_number += 1

            video_capture.release()

def add_frames_to_chromadb(video_dir, frames_dir):
    video_frames = {}

    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_title = video_file[:-4]
            frame_folder = os.path.join(frames_dir, video_title)
            if os.path.exists(frame_folder):
                video_frames[video_title] = [f for f in os.listdir(frame_folder) if f.endswith('.jpg')]

    ids = []
    uris = []
    metadatas = []

    for video_title, frames in video_frames.items():
        video_path = os.path.join(video_dir, f"{video_title}.mp4")
        for frame in frames:
            frame_id = f"{frame[:-4]}_{video_title}"
            frame_path = os.path.join(frames_dir, video_title, frame)
            ids.append(frame_id)
            uris.append(frame_path)
            metadatas.append({'video_uri': video_path})

    video_collection.add(ids=ids, uris=uris, metadatas=metadatas)

if audio_collection.count() == 0:
    with st.spinner("Indexing audio files... This may take a few minutes."):
        add_audio(audio_collection, audio_folder)
    st.success("Audio files indexed successfully!")

if image_collection.count() == 0:
    with st.spinner("Indexing image files... This may take a few minutes."):
        add_images(image_collection, image_folder)
    st.success("Image files indexed successfully!")

if text_collection.count() == 0:
    with st.spinner("Indexing text files... This may take a few minutes."):
        docs, embs, metas, ids = load_or_generate_wiki(wiki_embeddings_file)
        batch_add_to_collection(text_collection, docs, embs, metas, ids)
    st.success("Text files indexed successfully!")

if video_collection.count() == 0:
    with st.spinner("Extracting video frames and indexing... This may take a few minutes."):
        extract_frames(video_folder, video_frames_folder)
        add_frames_to_chromadb(video_folder, video_frames_folder)
    st.success("Video frames extracted and indexed successfully!")

st.header("Search Audio, Image, Text, and Video Files")
search_type = st.radio("Select search type:", ("Audio", "Image", "Text", "Video"))
query_text = st.text_input("Enter your search query:")
max_distance = st.slider("Max Distance", 0.0, 2.0, 1.5, 0.1)

if st.button("Search"):
    if search_type == "Audio":
        results = audio_collection.query(
            query_texts=[query_text],
            n_results=5,
            include=['uris', 'distances']
        )
        
        uris = results['uris'][0]
        distances = results['distances'][0]
        
        for uri, distance in zip(uris, distances):
            if distance <= max_distance:
                st.audio(uri)
                st.write(f"Distance: {distance:.2f}")
            else:
                st.write(f"Audio file filtered out (Distance: {distance:.2f})")
    
    elif search_type == "Image":
        results = image_collection.query(
            query_texts=[query_text],
            n_results=5,
            include=['uris', 'distances']
        )
        
        uris = results['uris'][0]
        distances = results['distances'][0]
        
        for uri, distance in zip(uris, distances):
            if distance <= max_distance:
                st.image(uri, caption=f"Distance: {distance:.2f}")
            else:
                st.write(f"Image file filtered out (Distance: {distance:.2f})")
    
    elif search_type == "Text":
        results = text_collection.query(
            query_texts=[query_text],
            n_results=5,
            include=['documents', 'distances', 'metadatas']
        )
        
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        for doc, distance, metadata in zip(documents, distances, metadatas):
            url = metadata.get('url', 'No URL available')
            wiki_id = metadata.get('wiki_id', 'Unknown Title')
            title = str(wiki_id).replace('_', ' ')
            if distance <= max_distance:
                st.markdown(f"**Title:** {title}")
                st.markdown(f"**Distance:** {distance:.2f}")
                st.markdown(f"**URL:** {url}")
                st.markdown(f"**Text:** {doc}\n")
            else:
                st.write(f"Text document filtered out - Title: {title} (Distance: {distance:.2f})")
    
    else:  # Video search
        results = video_collection.query(
            query_texts=[query_text],
            n_results=5,
            include=['uris', 'distances', 'metadatas']
        )
        
        uris = results['uris'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        displayed_videos = set()
        for uri, distance, metadata in zip(uris, distances, metadatas):
            video_uri = metadata['video_uri']
            if distance <= max_distance and video_uri not in displayed_videos:
                st.video(video_uri)
                st.write(f"Distance: {distance:.2f}")
                displayed_videos.add(video_uri)
            else:
                st.write(f"Video file filtered out (Distance: {distance:.2f})")

# Display some stats
st.sidebar.header("Collection Stats")
st.sidebar.write(f"Total audio files: {audio_collection.count()}")
st.sidebar.write(f"Total image files: {image_collection.count()}")
st.sidebar.write(f"Total text documents: {text_collection.count()}")
st.sidebar.write(f"Total video frames: {video_collection.count()}")
