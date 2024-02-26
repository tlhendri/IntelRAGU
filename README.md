# Intel Retrieval Augmented Generation (RAG) Utilities

An open-source initiave to document and share experiments to apply Retrieval Augmented Generation (RAG) techniques to Threat Intelligence searching capabilities.

Adapted from git at Cyb3rWard0g/IntelRAGU.

Made to be extensible from localized copy of intel store in custom directory. Details below. Model also localized for lightweight data transfer.

## Build Docker Image
Must be run from the IntelRAGU api directory [location of Dockerfile]
```
docker build . -t rag-chroma
```

## Define .ENV File

Create a `.env` file and define the `OPENAI_API_KEY` variable with your OpenAI Key. This is needed to use the [LangChain's ChatOpenAI module](https://python.langchain.com/docs/integrations/chat/openai). This is not needed to embed the ATT&CK Groups data. This is done with the [all-mpnet-base-v2 sentence-transformers model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) ;) . 

```
OPENAI_API_KEY=XXXXXXXXX
```

## Run Docker Image
Original:
```
docker run -it --rm --name rag-chroma --env-file .env -p 8080:8080 rag-chroma
```
New:
```
docker run -it --rm --name rag-chroma --env-file d:\intelRAGU\IntelRAGU\api\.env -p 8080:8080 --mount type=bind,src="[location of IntelRAGU]\api\custom",target="/custom" rag-chroma
```

On the first time of running the above command, the container will:
* Download the [all-mpnet-base-v2 sentence-transformers model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) (~400MB).
* Download the [Hugging Face Cyb3rWard0g/ATTCKGroups dataset](https://huggingface.co/datasets/Cyb3rWard0g/ATTCKGroups) (~846KB).
* Process the dataset by tokenizing and embedding every ATT&CK Group.
* Create the vector database by adding all the embeddings into a local [Chroma Database](https://www.trychroma.com).

The docker container will remove itself after it has been concluded. The image will persist.

On subsequent runs of the command, the docker will check for the localized versions of the transformer and intel set. The localized versions will be used. In the case of the transformer, a new pool will need to be set up for embedding. This may be quicker than the download.

Localize intel.csv takes the form of id, text, source. text will be enclosed in double quotes and should not contain double quotes. A piece of intel will exist on each line.

## Explore Playground

Browse to `http://127.0.0.1/rag-chroma/playground` and start asking questions.

![](images/LangServer-Playground.png)

## References
* https://python.langchain.com/docs/templates/
* https://python.langchain.com/docs/templates/rag-chroma
* https://huggingface.co/datasets/Cyb3rWard0g/ATTCKGroups