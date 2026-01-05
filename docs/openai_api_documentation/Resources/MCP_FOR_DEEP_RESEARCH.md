Building MCP servers for ChatGPT and API integrations

=====================================================



Build an MCP server to use with ChatGPT connectors, deep research, or API integrations.



\[Model Context Protocol](https://modelcontextprotocol.io/introduction) (MCP) is an open protocol that's becoming the industry standard for extending AI models with additional tools and knowledge. Remote MCP servers can be used to connect models over the Internet to new data sources and capabilities.



In this guide, we'll cover how to build a remote MCP server that reads data from a private data source (a \[vector store](/docs/guides/retrieval)) and makes it available in ChatGPT via connectors in chat and deep research, as well as \[via API](/docs/guides/deep-research).



\*\*Note\*\*: You can build and use full MCP connectors with the \*\*developer mode\*\* beta. Pro and Plus users can enable it under \*\*Settings → Connectors → Advanced → Developer mode\*\* to access the complete set of MCP tools. Learn more in the \[Developer mode guide](/docs/guides/developer-mode).



Configure a data source

-----------------------



You can use data from any source to power a remote MCP server, but for simplicity, we will use \[vector stores](/docs/guides/retrieval) in the OpenAI API. Begin by uploading a PDF document to a new vector store - \[you can use this public domain 19th century book about cats](https://cdn.openai.com/API/docs/cats.pdf) for an example.



You can upload files and create a vector store \[in the dashboard here](/storage/vector\_stores), or you can create vector stores and upload files via API. \[Follow the vector store guide](/docs/guides/retrieval) to set up a vector store and upload a file to it.



Make a note of the vector store's unique ID to use in the example to follow.



!\[vector store configuration](https://cdn.openai.com/API/docs/images/vector\_store.png)



Create an MCP server

--------------------



Next, let's create a remote MCP server that will do search queries against our vector store, and be able to return document content for files with a given ID.



In this example, we are going to build our MCP server using Python and \[FastMCP](https://github.com/jlowin/fastmcp). A full implementation of the server will be provided at the end of this section, along with instructions for running it on \[Replit](https://replit.com/).



Note that there are a number of other MCP server frameworks you can use in a variety of programming languages. Whichever framework you use though, the tool definitions in your server will need to conform to the shape described here.



To work with ChatGPT Connectors or deep research (in ChatGPT or via API), your MCP server must implement two tools - `search` and `fetch`.



\### `search` tool



The `search` tool is responsible for returning a list of relevant search results from your MCP server's data source, given a user's query.



\_Arguments:\_



A single query string.



\_Returns:\_



An object with a single key, `results`, whose value is an array of result objects. Each result object should include:



\*   `id` - a unique ID for the document or search result item

\*   `title` - human-readable title.

\*   `url` - canonical URL for citation.



In MCP, tool results must be returned as \[a content array](https://modelcontextprotocol.io/docs/learn/architecture#understanding-the-tool-execution-response) containing one or more "content items." Each content item has a type (such as `text`, `image`, or `resource`) and a payload.



For the `search` tool, you should return \*\*exactly one\*\* content item with:



\*   `type: "text"`

\*   `text`: a JSON-encoded string matching the results array schema above.



The final tool response should look like:



```

{

&nbsp; "content": \[

&nbsp;   {

&nbsp;     "type": "text",

&nbsp;     "text": "{\\"results\\":\[{\\"id\\":\\"doc-1\\",\\"title\\":\\"...\\",\\"url\\":\\"...\\"}]}"

&nbsp;   }

&nbsp; ]

}

```



\### `fetch` tool



The fetch tool is used to retrieve the full contents of a search result document or item.



\_Arguments:\_



A string which is a unique identifier for the search document.



\_Returns:\_



A single object with the following properties:



\*   `id` - a unique ID for the document or search result item

\*   `title` - a string title for the search result item

\*   `text` - The full text of the document or item

\*   `url` - a URL to the document or search result item. Useful for citing specific resources in research.

\*   `metadata` - an optional key/value pairing of data about the result



In MCP, tool results must be returned as \[a content array](https://modelcontextprotocol.io/docs/learn/architecture#understanding-the-tool-execution-response) containing one or more "content items." Each content item has a `type` (such as `text`, `image`, or `resource`) and a payload.



In this case, the `fetch` tool must return exactly \[one content item with `type: "text"`](https://modelcontextprotocol.io/specification/2025-06-18/server/tools#tool-result). The `text` field should be a JSON-encoded string of the document object following the schema above.



The final tool response should look like:



```

{

&nbsp; "content": \[

&nbsp;   {

&nbsp;     "type": "text",

&nbsp;     "text": "{\\"id\\":\\"doc-1\\",\\"title\\":\\"...\\",\\"text\\":\\"full text...\\",\\"url\\":\\"https://example.com/doc\\",\\"metadata\\":{\\"source\\":\\"vector\_store\\"}}"

&nbsp;   }

&nbsp; ]

}

```



\### Server example



An easy way to try out this example MCP server is using \[Replit](https://replit.com/). You can configure this sample application with your own API credentials and vector store information to try it yourself.



\[



Example MCP server on Replit



Remix the server example on Replit to test live.



](https://replit.com/@kwhinnery-oai/DeepResearchServer?v=1#README.md)



A full implementation of both the `search` and `fetch` tools in FastMCP is below also for convenience.



Full implementation - FastMCP server



```

"""

Sample MCP Server for ChatGPT Integration



This server implements the Model Context Protocol (MCP) with search and fetch

capabilities designed to work with ChatGPT's chat and deep research features.

"""



import logging

import os

from typing import Dict, List, Any



from fastmcp import FastMCP

from openai import OpenAI



\# Configure logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(\_\_name\_\_)



\# OpenAI configuration

OPENAI\_API\_KEY = os.environ.get("OPENAI\_API\_KEY")

VECTOR\_STORE\_ID = os.environ.get("VECTOR\_STORE\_ID", "")



\# Initialize OpenAI client

openai\_client = OpenAI()



server\_instructions = """

This MCP server provides search and document retrieval capabilities

for chat and deep research connectors. Use the search tool to find relevant documents

based on keywords, then use the fetch tool to retrieve complete

document content with citations.

"""



def create\_server():

&nbsp;   """Create and configure the MCP server with search and fetch tools."""



&nbsp;   # Initialize the FastMCP server

&nbsp;   mcp = FastMCP(name="Sample MCP Server",

&nbsp;                 instructions=server\_instructions)



&nbsp;   @mcp.tool()

&nbsp;   async def search(query: str) -> Dict\[str, List\[Dict\[str, Any]]]:

&nbsp;       """

&nbsp;       Search for documents using OpenAI Vector Store search.



&nbsp;       This tool searches through the vector store to find semantically relevant matches.

&nbsp;       Returns a list of search results with basic information. Use the fetch tool to get

&nbsp;       complete document content.



&nbsp;       Args:

&nbsp;           query: Search query string. Natural language queries work best for semantic search.



&nbsp;       Returns:

&nbsp;           Dictionary with 'results' key containing list of matching documents.

&nbsp;           Each result includes id, title, text snippet, and optional URL.

&nbsp;       """

&nbsp;       if not query or not query.strip():

&nbsp;           return {"results": \[]}



&nbsp;       if not openai\_client:

&nbsp;           logger.error("OpenAI client not initialized - API key missing")

&nbsp;           raise ValueError(

&nbsp;               "OpenAI API key is required for vector store search")



&nbsp;       # Search the vector store using OpenAI API

&nbsp;       logger.info(f"Searching {VECTOR\_STORE\_ID} for query: '{query}'")



&nbsp;       response = openai\_client.vector\_stores.search(

&nbsp;           vector\_store\_id=VECTOR\_STORE\_ID, query=query)



&nbsp;       results = \[]



&nbsp;       # Process the vector store search results

&nbsp;       if hasattr(response, 'data') and response.data:

&nbsp;           for i, item in enumerate(response.data):

&nbsp;               # Extract file\_id, filename, and content

&nbsp;               item\_id = getattr(item, 'file\_id', f"vs\_{i}")

&nbsp;               item\_filename = getattr(item, 'filename', f"Document {i+1}")



&nbsp;               # Extract text content from the content array

&nbsp;               content\_list = getattr(item, 'content', \[])

&nbsp;               text\_content = ""

&nbsp;               if content\_list and len(content\_list) > 0:

&nbsp;                   # Get text from the first content item

&nbsp;                   first\_content = content\_list\[0]

&nbsp;                   if hasattr(first\_content, 'text'):

&nbsp;                       text\_content = first\_content.text

&nbsp;                   elif isinstance(first\_content, dict):

&nbsp;                       text\_content = first\_content.get('text', '')



&nbsp;               if not text\_content:

&nbsp;                   text\_content = "No content available"



&nbsp;               # Create a snippet from content

&nbsp;               text\_snippet = text\_content\[:200] + "..." if len(

&nbsp;                   text\_content) > 200 else text\_content



&nbsp;               result = {

&nbsp;                   "id": item\_id,

&nbsp;                   "title": item\_filename,

&nbsp;                   "text": text\_snippet,

&nbsp;                   "url":

&nbsp;                   f"https://platform.openai.com/storage/files/{item\_id}"

&nbsp;               }



&nbsp;               results.append(result)



&nbsp;       logger.info(f"Vector store search returned {len(results)} results")

&nbsp;       return {"results": results}



&nbsp;   @mcp.tool()

&nbsp;   async def fetch(id: str) -> Dict\[str, Any]:

&nbsp;       """

&nbsp;       Retrieve complete document content by ID for detailed

&nbsp;       analysis and citation. This tool fetches the full document

&nbsp;       content from OpenAI Vector Store. Use this after finding

&nbsp;       relevant documents with the search tool to get complete

&nbsp;       information for analysis and proper citation.



&nbsp;       Args:

&nbsp;           id: File ID from vector store (file-xxx) or local document ID



&nbsp;       Returns:

&nbsp;           Complete document with id, title, full text content,

&nbsp;           optional URL, and metadata



&nbsp;       Raises:

&nbsp;           ValueError: If the specified ID is not found

&nbsp;       """

&nbsp;       if not id:

&nbsp;           raise ValueError("Document ID is required")



&nbsp;       if not openai\_client:

&nbsp;           logger.error("OpenAI client not initialized - API key missing")

&nbsp;           raise ValueError(

&nbsp;               "OpenAI API key is required for vector store file retrieval")



&nbsp;       logger.info(f"Fetching content from vector store for file ID: {id}")



&nbsp;       # Fetch file content from vector store

&nbsp;       content\_response = openai\_client.vector\_stores.files.content(

&nbsp;           vector\_store\_id=VECTOR\_STORE\_ID, file\_id=id)



&nbsp;       # Get file metadata

&nbsp;       file\_info = openai\_client.vector\_stores.files.retrieve(

&nbsp;           vector\_store\_id=VECTOR\_STORE\_ID, file\_id=id)



&nbsp;       # Extract content from paginated response

&nbsp;       file\_content = ""

&nbsp;       if hasattr(content\_response, 'data') and content\_response.data:

&nbsp;           # Combine all content chunks from FileContentResponse objects

&nbsp;           content\_parts = \[]

&nbsp;           for content\_item in content\_response.data:

&nbsp;               if hasattr(content\_item, 'text'):

&nbsp;                   content\_parts.append(content\_item.text)

&nbsp;           file\_content = "\\n".join(content\_parts)

&nbsp;       else:

&nbsp;           file\_content = "No content available"



&nbsp;       # Use filename as title and create proper URL for citations

&nbsp;       filename = getattr(file\_info, 'filename', f"Document {id}")



&nbsp;       result = {

&nbsp;           "id": id,

&nbsp;           "title": filename,

&nbsp;           "text": file\_content,

&nbsp;           "url": f"https://platform.openai.com/storage/files/{id}",

&nbsp;           "metadata": None

&nbsp;       }



&nbsp;       # Add metadata if available from file info

&nbsp;       if hasattr(file\_info, 'attributes') and file\_info.attributes:

&nbsp;           result\["metadata"] = file\_info.attributes



&nbsp;       logger.info(f"Fetched vector store file: {id}")

&nbsp;       return result



&nbsp;   return mcp



def main():

&nbsp;   """Main function to start the MCP server."""

&nbsp;   # Verify OpenAI client is initialized

&nbsp;   if not openai\_client:

&nbsp;       logger.error(

&nbsp;           "OpenAI API key not found. Please set OPENAI\_API\_KEY environment variable."

&nbsp;       )

&nbsp;       raise ValueError("OpenAI API key is required")



&nbsp;   logger.info(f"Using vector store: {VECTOR\_STORE\_ID}")



&nbsp;   # Create the MCP server

&nbsp;   server = create\_server()



&nbsp;   # Configure and start the server

&nbsp;   logger.info("Starting MCP server on 0.0.0.0:8000")

&nbsp;   logger.info("Server will be accessible via SSE transport")



&nbsp;   try:

&nbsp;       # Use FastMCP's built-in run method with SSE transport

&nbsp;       server.run(transport="sse", host="0.0.0.0", port=8000)

&nbsp;   except KeyboardInterrupt:

&nbsp;       logger.info("Server stopped by user")

&nbsp;   except Exception as e:

&nbsp;       logger.error(f"Server error: {e}")

&nbsp;       raise



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   main()

```



Replit setup



On Replit, you will need to configure two environment variables in the "Secrets" UI:



\*   `OPENAI\_API\_KEY` - Your standard OpenAI API key

\*   `VECTOR\_STORE\_ID` - The unique identifier of a vector store that can be used for search - the one you created earlier.



On free Replit accounts, server URLs are active for as long as the editor is active, so while you are testing, you'll need to keep the browser tab open. You can get a URL for your MCP server by clicking on the chainlink icon:



!\[replit configuration](https://cdn.openai.com/API/docs/images/replit.png)



In the long dev URL, ensure it ends with `/sse/`, which is the server-sent events (streaming) interface to the MCP server. This is the URL you will use to import your connector both via API and ChatGPT. An example Replit URL looks like:



```

https://777xxx.janeway.replit.dev/sse/

```



Test and connect your MCP server

--------------------------------



You can test your MCP server with a deep research model \[in the prompts dashboard](/chat). Create a new prompt, or edit an existing one, and add a new MCP tool to the prompt configuration. Remember that MCP servers used via API for deep research have to be configured with no approval required.



!\[prompts configuration](https://cdn.openai.com/API/docs/images/prompts\_mcp.png)



Once you have configured your MCP server, you can chat with a model using it via the Prompts UI.



!\[prompts chat](https://cdn.openai.com/API/docs/images/chat\_prompts\_mcp.png)



You can test the MCP server using the Responses API directly with a request like this one:



```

curl https://api.openai.com/v1/responses \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp; -d '{

&nbsp; "model": "o4-mini-deep-research",

&nbsp; "input": \[

&nbsp;   {

&nbsp;     "role": "developer",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "input\_text",

&nbsp;         "text": "You are a research assistant that searches MCP servers to find answers to your questions."

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   {

&nbsp;     "role": "user",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "input\_text",

&nbsp;         "text": "Are cats attached to their homes? Give a succinct one page overview."

&nbsp;       }

&nbsp;     ]

&nbsp;   }

&nbsp; ],

&nbsp; "reasoning": {

&nbsp;   "summary": "auto"

&nbsp; },

&nbsp; "tools": \[

&nbsp;   {

&nbsp;     "type": "mcp",

&nbsp;     "server\_label": "cats",

&nbsp;     "server\_url": "https://777ff573-9947-4b9c-8982-658fa40c7d09-00-3le96u7wsymx.janeway.replit.dev/sse/",

&nbsp;     "allowed\_tools": \[

&nbsp;       "search",

&nbsp;       "fetch"

&nbsp;     ],

&nbsp;     "require\_approval": "never"

&nbsp;   }

&nbsp; ]

}'

```



\### Handle authentication



As someone building a custom remote MCP server, authorization and authentication help you protect your data. We recommend using OAuth and \[dynamic client registration](https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization#2-4-dynamic-client-registration). To learn more about the protocol's authentication, read the \[MCP user guide](https://modelcontextprotocol.io/docs/concepts/transports#authentication-and-authorization) or see the \[authorization specification](https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization).



If you connect your custom remote MCP server in ChatGPT, users in your workspace will get an OAuth flow to your application.



\### Connect in ChatGPT



1\.  Import your remote MCP servers directly in \[ChatGPT settings](https://chatgpt.com/#settings).

2\.  Connect your server in the \*\*Connectors\*\* tab. It should now be visible in the composer's "Deep Research" and "Use Connectors" tools. You may have to add the server as a source.

3\.  Test your server by running some prompts.



Risks and safety

----------------



Custom MCP servers enable you to connect your ChatGPT workspace to external applications, which allows ChatGPT to access, send and receive data in these applications. Please note that custom MCP servers are not developed or verified by OpenAI, and are third-party services that are subject to their own terms and conditions.



If you come across a malicious MCP server, please report it to \[security@openai.com](/docs/mcp).



\### Prompt injection-related risks



Prompt injections are a form of attack where an attacker embeds malicious instructions in content that one of our models is likely to encounter–such as a webpage–with the intention that the instructions override ChatGPT’s intended behavior. If the model obeys the injected instructions it may take actions the user and developer never intended—including sending private data to an external destination.



For example, you might ask ChatGPT to find a restaurant for a group dinner by checking your calendar and recent emails. While researching, it might encounter a malicious comment—essentially a harmful piece of content designed to trick the agent into performing unintended actions—directing it to retrieve a password reset code from Gmail and send it to a malicious website.



Below is a table of specific scenarios to consider. We recommend reviewing this table carefully to inform your decision about whether to use custom MCPs.



|Scenario / Risk|Is it safe if I trust the MCP’s developer?|What can I do to reduce risk?|

|---|---|---|

|An attacker may somehow insert a prompt injection attack into data accessible via the MCP.Examples:• For a customer support MCP, an attacker could send you a customer support request with a prompt injection attack.|Trusting a MCP’s developer does not make this safe.For this to be safe you need to trust all content that can be accessed within the MCP.|• Do not use a MCP if it could contain malicious or untrusted user input, even if you trust the developer of the MCP.• Configure access to minimize how many people have access to the MCP.|

|A malicious MCP may request excessive parameters to a read or write action.Example:• An employee flight booking MCP could expose a read action to get a flight schedule, but request parameters including summaryOfConversation, userAnnualIncome, userHomeAddress.|Trusting a MCP’s developer does not necessarily make this safe.A MCP’s developer may consider it reasonable to be requesting certain data that you do not consider acceptable to share.|• When sideloading MCPs, carefully review the parameters being requested for each action and ensure there is no privacy overreach.|

|An attacker may use a prompt injection attack to trick ChatGPT into fetching sensitive data from a custom MCP, to then be sent to the attacker.Example:• An attacker may deliver a prompt injection attack to one of the enterprise users via a different MCP (e.g. for email), where the attack attempts to trick ChatGPT into reading sensitive data from some internal tool MCP and then attempt to exfiltrate it.|Trusting a MCP’s developer does not make this safe.Everything within the new MCP could be safe and trusted since the risk is this data being stolen by attacks coming from a different malicious source.|• ChatGPT is designed to protect users, but attackers may attempt to steal your data, so be aware of the risk and consider whether taking it makes sense.• Configure access to minimize how many people have access to MCPs with particularly sensitive data.|

|An attacker may use a prompt injection attack to exfiltrate sensitive information through a write action to a custom MCP.Example:• An attacker uses a prompt injection attack (via a different MCP) to trick ChatGPT into fetching sensitive data, and then exfiltrates it by tricking ChatGPT into using a MCP for a customer support system to send it to the attacker.|Trusting a MCP’s developer does not make this safe.Even if you fully trust the MCP, if write actions have any consequences that can be observed by an attacker, they could attempt to take advantage of it.|• Users should review write actions carefully when they happen (to ensure they were intended and do not contain any data that shouldn’t be shared).|

|An attacker may use a prompt injection attack to exfiltrate sensitive information through a read action to a malicious custom MCP (since these can be logged by the MCP).|This attack only works if the MCP is malicious, or if the MCP incorrectly marks write actions as read actions.If you trust a MCP’s developer to correctly only mark read actions as read, and trust that developer to not attempt to steal data, then this risk is likely minimal.|• Only use MCPs from developers that you trust (though note this isn’t sufficient to make it safe).|

|An attacker may use a prompt injection attack to trick ChatGPT into taking a harmful or destructive write action via a custom MCP that users did not intend.|Trusting a MCP’s developer does not make this safe.Everything within the new MCP could be safe and trusted, and this risk still exists since the attack comes from a different malicious source.|• Users should carefully review write actions to ensure they are intended and correct.• ChatGPT is designed to protect users, but attackers may attempt to trick ChatGPT into taking unintended write actions.• Configure access to minimize how many people have access to MCPs with particularly sensitive data.|



\### Non-prompt injection related risks



There are additional risks of custom MCPs, unrelated to prompt injection attacks:



\*   \*\*Write actions can increase both the usefulness and the risks of MCP servers\*\*, because they make it possible for the server to take potentially destructive actions rather than simply providing information back to ChatGPT. ChatGPT currently requires manual confirmation in any conversation before write actions can be taken. The confirmation will flag potentially sensitive data but you should only use write actions in situations where you have carefully considered, and are comfortable with, the possibility that ChatGPT might make a mistake involving such an action. It is possible for write actions to occur even if the MCP server has tagged the action as read only, making it even more important that you trust the custom MCP server before deploying to ChatGPT.

\*   \*\*Any MCP server may receive sensitive data as part of querying\*\*. Even when the server is not malicious, it will have access to whatever data ChatGPT supplies during the interaction, potentially including sensitive data the user may earlier have provided to ChatGPT. For instance, such data could be included in queries ChatGPT sends to the MCP server when using deep research or chat connectors.



\### Connecting to trusted servers



We recommend that you do not connect to a custom MCP server unless you know and trust the underlying application.



For example, always pick official servers hosted by the service providers themselves (e.g., connect to the Stripe server hosted by Stripe themselves on mcp.stripe.com, instead of an unofficial Stripe MCP server hosted by a third party). Because there aren't many official MCP servers today, you may be tempted to use a MCP server hosted by an organization that doesn't operate that server and simply proxies requests to that service via an API. This is not recommended—and you should only connect to an MCP once you’ve carefully reviewed how they use your data and have verified that you can trust the server. When building and connecting to your own MCP server, double check that it's the correct server. Be very careful with which data you provide in response to requests to your MCP server, and with how you treat the data sent to you as part of OpenAI calling your MCP server.



Your remote MCP server permits others to connect OpenAI to your services and allows OpenAI to access, send and receive data, and take action in these services. Avoid putting any sensitive information in the JSON for your tools, and avoid storing any sensitive information from ChatGPT users accessing your remote MCP server.



As someone building an MCP server, don't put anything malicious in your tool definitions.



Was this page useful?

