from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Настройки RAG
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
KNOWLEDGE_FILE = BASE_DIR / "knowledge.txt"
COLLECTION_NAME = "ru_knowledge_base"

# Базовый системный промпт
BASE_SYSTEM_PROMPT = (
    "Ты полезный ассистент. Отвечай на русском языке простым текстом. "
    "НЕ ИСПОЛЬЗУЙ Markdown (никаких звездочек **, решеток # или обратных кавычек). "
    "Пиши обычным текстом, используй абзацы и тире для списков, если это нужно.\n\n"
    "Ниже предоставлен контекст из базы знаний. Используй его, чтобы ответить на вопрос пользователя. "
    "Если ответа нет в контексте, честно скажи, что не знаешь, но можешь попробовать ответить из общих знаний.\n\n"
    "КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{context}\n"
)

def get_embeddings():
    """Создает русскоязычную embedding-функцию (модель)."""
    # Модель e5-small отлично работает с русским языком и занимает немного места.
    # При первом запуске она скачается с HuggingFace (около 400 МБ).
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': 'cpu'} # Если есть видеокарта, можно поменять на 'cuda'
    )

def build_or_load_db():
    """Создает базу данных (если её нет) или загружает существующую."""
    embeddings = get_embeddings()

    # Загружаем готовую непустую коллекцию. Папка может остаться после
    # прерванной первой индексации, поэтому проверяем именно коллекцию.
    if Path(CHROMA_DIR).exists():
        print("Загрузка существующей базы знаний...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        if vectorstore._collection.count() > 0:
            return vectorstore
        print("Коллекция пуста, индексирую базу знаний заново...")

    if not KNOWLEDGE_FILE.is_file():
        raise FileNotFoundError(f"Файл базы знаний не найден: {KNOWLEDGE_FILE}")

    print("Создание новой базы знаний...")
    # 1. Загружаем текст
    loader = TextLoader(KNOWLEDGE_FILE, encoding="utf-8")
    docs = loader.load()

    # 2. Умный сплиттер (разбиваем текст на чанки)
    # Разбивает сначала по абзацам, потом по предложениям, чтобы не рвать смысл
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Размер куска
        chunk_overlap=150,    # Перекрытие, чтобы не терять контекст на стыках
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""] # Учитываем русскую пунктуацию
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Текст разбит на {len(chunks)} чанков.")

    # 3. Создаем коллекцию в БД и добавляем туда документы
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )
    print("База знаний успешно создана!")
    return vectorstore

def get_context_and_prompt(query, vectorstore):
    """Ищет контекст по запросу и формирует итоговый системный промпт."""
    context_text = "В базе знаний нет информации по этому вопросу."
    
    if vectorstore is not None:
        # Ищем 3 самых релевантных чанка
        docs = vectorstore.similarity_search(f"query: {query}", k=3)
        if docs:
            context_text = "\n\n---\n\n".join([d.page_content for d in docs])
            print("Найден контекст в базе знаний!")

    # Подставляем найденный контекст в наш базовый промпт
    final_prompt = BASE_SYSTEM_PROMPT.format(context=context_text)
    return final_prompt
