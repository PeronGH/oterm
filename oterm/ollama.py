from typing import Any, AsyncGenerator, AsyncIterator, Literal

from ollama import AsyncClient

from oterm.config import envConfig


class OllamaLLM:

    def __init__(
        self,
        model="nous-hermes:13b",
        system: str | None = None,
        context: list[int] = [],
        format: Literal["", "json"] = "",
    ):
        self.model = model
        self.system = system
        self.context = context
        self.format = format

    def __client(self) -> AsyncClient:
        return AsyncClient(
            host=envConfig.OLLAMA_URL, verify=envConfig.OTERM_VERIFY_SSL, http2=True
        )

    async def completion(self, prompt: str, images: list[str] = []) -> str:
        async with self.__client() as client:
            response: dict = await client.generate(
                model=self.model,
                prompt=prompt,
                context=self.context,
                system=self.system,  # type: ignore
                format=self.format,  # type: ignore
                images=images,
            )
            self.context = response.get("context", [])
            return response.get("response", "")

    async def stream(
        self, prompt: str, images: list[str] = []
    ) -> AsyncGenerator[str, Any]:
        async with self.__client() as client:
            stream: AsyncIterator[dict] = await client.generate(
                model=self.model,
                prompt=prompt,
                context=self.context,
                system=self.system,  # type: ignore
                format=self.format,  # type: ignore
                images=images,
                stream=True,
            )
            text = ""
            async for response in stream:
                text = text + response.get("response", "")
                if "context" in response:
                    self.context = response.get("context")
                yield text
