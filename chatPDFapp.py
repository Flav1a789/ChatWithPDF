import tempfile
from PIL import Image

# Importer le module os pour définir la clé API
# Import os to set API key
import os

# Importer OpenAI comme principal service LLM
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

# Importer streamlit pour l'interface utilisateur/application
# Bring in streamlit for UI/app interface
import streamlit as st

# Importer les chargeurs de documents PDF... il en existe d'autres !
# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader

# Importer chroma comme magasin de vecteurs
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Importer les outils liés aux magasins de vecteurs
# Import vector store related tools
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Définir le titre et le sous-titre de l'application
# Set the title and subtitle of the app
st.title('🦜🔗 PDF-Chat: Interagissez avec vos PDFs de manière conversationnelle')
st.subheader('Chargez votre PDF, posez des questions et recevez des réponses directement du document.')

# Charger l'image de l'interface
# Load the interface image
image = Image.open('PDF-Chat App.png')
st.image(image)

# Charger le fichier PDF et retourner un chemin temporaire
# Load the PDF file and return a temporary path for it
st.subheader('Téléchargez votre PDF')
uploaded_file = st.file_uploader('', type=(['pdf', 'tsv', 'csv', 'txt', 'tab', 'xlsx', 'xls']))

# Chemin temporaire pour stocker le fichier
# Temporary path to store the file
temp_file_path = os.getcwd()

# Boucle jusqu'à ce qu'un fichier soit chargé
# Loop until a file is uploaded
while uploaded_file is None:
    x = 1

if uploaded_file is not None:
    # Sauvegarder le fichier téléchargé dans un emplacement temporaire
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # Afficher le chemin complet du fichier téléchargé
    # Display the full path of the uploaded file
    st.write("Chemin complet du fichier téléchargé :", temp_file_path)

# Définir la clé API pour le service OpenAI
# Set API key for OpenAI service
os.environ['OPENAI_API_KEY'] = # Votre clé API OpenAI

# Créer une instance du modèle OpenAI LLM
# Create an instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Créer et charger le chargeur PDF
# Create and load PDF Loader
loader = PyPDFLoader(temp_file_path)

# Diviser les pages du PDF
# Split pages from PDF
pages = loader.load_and_split()

# Charger les documents dans la base de données vectorielle, alias ChromaDB
# Load documents into the vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='Pdf')

# Créer un objet d'information sur le magasin de vecteurs
# Create a vector store info object
vectorstore_info = VectorStoreInfo(
    name="Pdf",
    description="Un fichier PDF pour répondre à vos questions",
    vectorstore=store
)

# Convertir le magasin de documents en un kit d'outils Langchain
# Convert the document store into a Langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Ajouter le kit d'outils à un agent de Langchain de bout en bout
# Add the toolkit to an end-to-end Langchain agent
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# Créer une zone de saisie de texte pour l'utilisateur
# Create a text input box for the user
prompt = st.text_input('Entrez votre question ici')

# Si l'utilisateur appuie sur entrée
# If the user hits enter
if prompt:
    # Envoyer la question au modèle LLM
    # Pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # Afficher la réponse à l'écran
    # Display the response on the screen
    st.write(response)

    # Avec un expander Streamlit
    # With a Streamlit expander
    with st.expander('Recherche de similarité dans le document'):
        # Trouver les pages pertinentes
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt)
        # Afficher le contenu de la première page trouvée
        # Display the content of the first page found
        st.write(search[0][0].page_content)
