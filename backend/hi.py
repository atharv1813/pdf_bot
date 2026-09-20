from dotenv import load_dotenv
load_dotenv()

from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(
    model="amazon.nova-lite-v1:0",
    region_name="us-east-1"
)

print(llm.invoke("Hello").content)