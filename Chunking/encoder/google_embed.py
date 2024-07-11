import os
from typing import Any, List, Optional

from encoder import BaseEncoder

class EncoderInfo(BaseModel):
    name: str
    token_limit: int
    threshold: Optional[float] = None


model_configs = {
    "textembedding-gecko@001": EncoderInfo(
        name="embedding-gecko-001",
        token_limit=1024,
        threshold=0.7,
    ),
    "textembedding-gecko@003": EncoderInfo(
        name="embedding-gecko-003",
        token_limit=1024,
        threshold=0.7,
    ),
    "text-embedding-004": EncoderInfo(
        name="text-embedding-004",
        token_limit=2048,
        threshold=0.7,
    ),
}

class GoogleEncoder(BaseEncoder):
    """GoogleEncoder class for generating embeddings using Google's AI Platform.

    Attributes:
        client: An instance of the TextEmbeddingModel client.
        type: The type of the encoder, which is "google".
    """

    client: Optional[Any] = None
    type: str = "google"

    def __init__(
        self,
        name: Optional[str] = None,
        score_threshold: float = 0.75,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        if name is None:
            name = "textembedding-gecko@003"

        super().__init__(name=name, score_threshold=score_threshold)

        self.client = self._initialize_client(project_id, location, api_endpoint)

    def _initialize_client(self, project_id, location, api_endpoint):
        try:
            from google.cloud import aiplatform
            from vertexai.language_models import TextEmbeddingModel
        except ImportError:
            raise ImportError(
                "Please install Google Cloud and Vertex AI libraries to use GoogleEncoder. "
                "You can install them with: "
                "`pip install google-cloud-aiplatform vertexai`"
            )

        project_id = project_id or os.getenv("GOOGLE_PROJECT_ID")
        location = location or os.getenv("GOOGLE_LOCATION", "us-central1")
        api_endpoint = api_endpoint or os.getenv("GOOGLE_API_ENDPOINT")

        if project_id is None:
            raise ValueError("Google Project ID cannot be 'None'.")

        try:
            aiplatform.init(
                project=project_id, location=location, endpoint=api_endpoint
            )
            client = TextEmbeddingModel.from_pretrained(self.name)
        except Exception as err:
            raise ValueError(
                f"Google AI Platform client failed to initialize. Error: {err}"
            ) from err

        return client

    def __call__(self, docs: List[str]) -> List[List[float]]:
        if self.client is None:
            raise ValueError("Google AI Platform client is not initialized.")
        try:
            embeddings = self.client.get_embeddings(docs)
            return [embedding.values for embedding in embeddings]
        except Exception as e:
            raise ValueError(f"Google AI Platform API call failed. Error: {e}") from e