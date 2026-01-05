File search

===========



Allow models to search your files for relevant information before generating a response.



File search is a tool available in the \[Responses API](/docs/api-reference/responses). It enables models to retrieve information in a knowledge base of previously uploaded files through semantic and keyword search. By creating vector stores and uploading files to them, you can augment the models' inherent knowledge by giving them access to these knowledge bases or `vector\_stores`.



To learn more about how vector stores and semantic search work, refer to our \[retrieval guide](/docs/guides/retrieval).



This is a hosted tool managed by OpenAI, meaning you don't have to implement code on your end to handle its execution. When the model decides to use it, it will automatically call the tool, retrieve information from your files, and return an output.



How to use

----------



Prior to using file search with the Responses API, you need to have set up a knowledge base in a vector store and uploaded files to it.



Create a vector store and upload a file



Follow these steps to create a vector store and upload a file to it. You can use \[this example file](https://cdn.openai.com/API/docs/deep\_research\_blog.pdf) or upload your own.



\#### Upload the file to the File API



Upload a file



```

import requests

from io import BytesIO

from openai import OpenAI



client = OpenAI()



def create\_file(client, file\_path):

&nbsp;   if file\_path.startswith("http://") or file\_path.startswith("https://"):

&nbsp;       # Download the file content from the URL

&nbsp;       response = requests.get(file\_path)

&nbsp;       file\_content = BytesIO(response.content)

&nbsp;       file\_name = file\_path.split("/")\[-1]

&nbsp;       file\_tuple = (file\_name, file\_content)

&nbsp;       result = client.files.create(

&nbsp;           file=file\_tuple,

&nbsp;           purpose="assistants"

&nbsp;       )

&nbsp;   else:

&nbsp;       # Handle local file path

&nbsp;       with open(file\_path, "rb") as file\_content:

&nbsp;           result = client.files.create(

&nbsp;               file=file\_content,

&nbsp;               purpose="assistants"

&nbsp;           )

&nbsp;   print(result.id)

&nbsp;   return result.id



\# Replace with your own file path or URL

file\_id = create\_file(client, "https://cdn.openai.com/API/docs/deep\_research\_blog.pdf")

```



```

import fs from "fs";

import OpenAI from "openai";

const openai = new OpenAI();



async function createFile(filePath) {

&nbsp; let result;

&nbsp; if (filePath.startsWith("http://") || filePath.startsWith("https://")) {

&nbsp;   // Download the file content from the URL

&nbsp;   const res = await fetch(filePath);

&nbsp;   const buffer = await res.arrayBuffer();

&nbsp;   const urlParts = filePath.split("/");

&nbsp;   const fileName = urlParts\[urlParts.length - 1];

&nbsp;   const file = new File(\[buffer], fileName);

&nbsp;   result = await openai.files.create({

&nbsp;     file: file,

&nbsp;     purpose: "assistants",

&nbsp;   });

&nbsp; } else {

&nbsp;   // Handle local file path

&nbsp;   const fileContent = fs.createReadStream(filePath);

&nbsp;   result = await openai.files.create({

&nbsp;     file: fileContent,

&nbsp;     purpose: "assistants",

&nbsp;   });

&nbsp; }

&nbsp; return result.id;

}



// Replace with your own file path or URL

const fileId = await createFile(

&nbsp; "https://cdn.openai.com/API/docs/deep\_research\_blog.pdf"

);



console.log(fileId);

```



\#### Create a vector store



Create a vector store



```

vector\_store = client.vector\_stores.create(

&nbsp;   name="knowledge\_base"

)

print(vector\_store.id)

```



```

const vectorStore = await openai.vectorStores.create({

&nbsp;   name: "knowledge\_base",

});

console.log(vectorStore.id);

```



\#### Add the file to the vector store



Add a file to a vector store



```

result = client.vector\_stores.files.create(

&nbsp;   vector\_store\_id=vector\_store.id,

&nbsp;   file\_id=file\_id

)

print(result)

```



```

await openai.vectorStores.files.create(

&nbsp;   vectorStore.id,

&nbsp;   {

&nbsp;       file\_id: fileId,

&nbsp;   }

});

```



\#### Check status



Run this code until the file is ready to be used (i.e., when the status is `completed`).



Check status



```

result = client.vector\_stores.files.list(

&nbsp;   vector\_store\_id=vector\_store.id

)

print(result)

```



```

const result = await openai.vectorStores.files.list({

&nbsp;   vector\_store\_id: vectorStore.id,

});

console.log(result);

```



Once your knowledge base is set up, you can include the `file\_search` tool in the list of tools available to the model, along with the list of vector stores in which to search.



File search tool



```

from openai import OpenAI

client = OpenAI()



response = client.responses.create(

&nbsp;   model="gpt-4.1",

&nbsp;   input="What is deep research by OpenAI?",

&nbsp;   tools=\[{

&nbsp;       "type": "file\_search",

&nbsp;       "vector\_store\_ids": \["<vector\_store\_id>"]

&nbsp;   }]

)

print(response)

```



```

import OpenAI from "openai";

const openai = new OpenAI();



const response = await openai.responses.create({

&nbsp;   model: "gpt-4.1",

&nbsp;   input: "What is deep research by OpenAI?",

&nbsp;   tools: \[

&nbsp;       {

&nbsp;           type: "file\_search",

&nbsp;           vector\_store\_ids: \["<vector\_store\_id>"],

&nbsp;       },

&nbsp;   ],

});

console.log(response);

```



```

using OpenAI.Responses;



string key = Environment.GetEnvironmentVariable("OPENAI\_API\_KEY")!;

OpenAIResponseClient client = new(model: "gpt-5", apiKey: key);



ResponseCreationOptions options = new();

options.Tools.Add(ResponseTool.CreateFileSearchTool(\["<vector\_store\_id>"]));



OpenAIResponse response = (OpenAIResponse)client.CreateResponse(\[

&nbsp;   ResponseItem.CreateUserMessageItem(\[

&nbsp;       ResponseContentPart.CreateInputTextPart("What is deep research by OpenAI?"),

&nbsp;   ]),

], options);



Console.WriteLine(response.GetOutputText());

```



When this tool is called by the model, you will receive a response with multiple outputs:



1\.  A `file\_search\_call` output item, which contains the id of the file search call.

2\.  A `message` output item, which contains the response from the model, along with the file citations.



File search response



```

{

&nbsp; "output": \[

&nbsp;   {

&nbsp;     "type": "file\_search\_call",

&nbsp;     "id": "fs\_67c09ccea8c48191ade9367e3ba71515",

&nbsp;     "status": "completed",

&nbsp;     "queries": \["What is deep research?"],

&nbsp;     "search\_results": null

&nbsp;   },

&nbsp;   {

&nbsp;     "id": "msg\_67c09cd3091c819185af2be5d13d87de",

&nbsp;     "type": "message",

&nbsp;     "role": "assistant",

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "output\_text",

&nbsp;         "text": "Deep research is a sophisticated capability that allows for extensive inquiry and synthesis of information across various domains. It is designed to conduct multi-step research tasks, gather data from multiple online sources, and provide comprehensive reports similar to what a research analyst would produce. This functionality is particularly useful in fields requiring detailed and accurate information...",

&nbsp;         "annotations": \[

&nbsp;           {

&nbsp;             "type": "file\_citation",

&nbsp;             "index": 992,

&nbsp;             "file\_id": "file-2dtbBZdjtDKS8eqWxqbgDi",

&nbsp;             "filename": "deep\_research\_blog.pdf"

&nbsp;           },

&nbsp;           {

&nbsp;             "type": "file\_citation",

&nbsp;             "index": 992,

&nbsp;             "file\_id": "file-2dtbBZdjtDKS8eqWxqbgDi",

&nbsp;             "filename": "deep\_research\_blog.pdf"

&nbsp;           },

&nbsp;           {

&nbsp;             "type": "file\_citation",

&nbsp;             "index": 1176,

&nbsp;             "file\_id": "file-2dtbBZdjtDKS8eqWxqbgDi",

&nbsp;             "filename": "deep\_research\_blog.pdf"

&nbsp;           },

&nbsp;           {

&nbsp;             "type": "file\_citation",

&nbsp;             "index": 1176,

&nbsp;             "file\_id": "file-2dtbBZdjtDKS8eqWxqbgDi",

&nbsp;             "filename": "deep\_research\_blog.pdf"

&nbsp;           }

&nbsp;         ]

&nbsp;       }

&nbsp;     ]

&nbsp;   }

&nbsp; ]

}

```



Retrieval customization

-----------------------



\### Limiting the number of results



Using the file search tool with the Responses API, you can customize the number of results you want to retrieve from the vector stores. This can help reduce both token usage and latency, but may come at the cost of reduced answer quality.



Limit the number of results



```

response = client.responses.create(

&nbsp;   model="gpt-4.1",

&nbsp;   input="What is deep research by OpenAI?",

&nbsp;   tools=\[{

&nbsp;       "type": "file\_search",

&nbsp;       "vector\_store\_ids": \["<vector\_store\_id>"],

&nbsp;       "max\_num\_results": 2

&nbsp;   }]

)

print(response)

```



```

const response = await openai.responses.create({

&nbsp;   model: "gpt-4.1",

&nbsp;   input: "What is deep research by OpenAI?",

&nbsp;   tools: \[{

&nbsp;       type: "file\_search",

&nbsp;       vector\_store\_ids: \["<vector\_store\_id>"],

&nbsp;       max\_num\_results: 2,

&nbsp;   }],

});

console.log(response);

```



\### Include search results in the response



While you can see annotations (references to files) in the output text, the file search call will not return search results by default.



To include search results in the response, you can use the `include` parameter when creating the response.



Include search results



```

response = client.responses.create(

&nbsp;   model="gpt-4.1",

&nbsp;   input="What is deep research by OpenAI?",

&nbsp;   tools=\[{

&nbsp;       "type": "file\_search",

&nbsp;       "vector\_store\_ids": \["<vector\_store\_id>"]

&nbsp;   }],

&nbsp;   include=\["file\_search\_call.results"]

)

print(response)

```



```

const response = await openai.responses.create({

&nbsp;   model: "gpt-4.1",

&nbsp;   input: "What is deep research by OpenAI?",

&nbsp;   tools: \[{

&nbsp;       type: "file\_search",

&nbsp;       vector\_store\_ids: \["<vector\_store\_id>"],

&nbsp;   }],

&nbsp;   include: \["file\_search\_call.results"],

});

console.log(response);

```



\### Metadata filtering



You can filter the search results based on the metadata of the files. For more details, refer to our \[retrieval guide](/docs/guides/retrieval), which covers:



\*   How to \[set attributes on vector store files](/docs/guides/retrieval#attributes)

\*   How to \[define filters](/docs/guides/retrieval#attribute-filtering)



Metadata filtering



```

response = client.responses.create(

&nbsp;   model="gpt-4.1",

&nbsp;   input="What is deep research by OpenAI?",

&nbsp;   tools=\[{

&nbsp;       "type": "file\_search",

&nbsp;       "vector\_store\_ids": \["<vector\_store\_id>"],

&nbsp;       "filters": {

&nbsp;           "type": "in",

&nbsp;           "key": "category",

&nbsp;           "value": \["blog", "announcement"]

&nbsp;       }

&nbsp;   }]

)

print(response)

```



```

const response = await openai.responses.create({

&nbsp;   model: "gpt-4.1",

&nbsp;   input: "What is deep research by OpenAI?",

&nbsp;   tools: \[{

&nbsp;       type: "file\_search",

&nbsp;       vector\_store\_ids: \["<vector\_store\_id>"],

&nbsp;       filters: {

&nbsp;           type: "in",

&nbsp;           key: "category",

&nbsp;           value: \["blog", "announcement"]

&nbsp;       }

&nbsp;   }]

});

console.log(response);

```



Supported files

---------------



\_For `text/` MIME types, the encoding must be one of `utf-8`, `utf-16`, or `ascii`.\_



|File format|MIME type|

|---|---|

|.c|text/x-c|

|.cpp|text/x-c++|

|.cs|text/x-csharp|

|.css|text/css|

|.doc|application/msword|

|.docx|application/vnd.openxmlformats-officedocument.wordprocessingml.document|

|.go|text/x-golang|

|.html|text/html|

|.java|text/x-java|

|.js|text/javascript|

|.json|application/json|

|.md|text/markdown|

|.pdf|application/pdf|

|.php|text/x-php|

|.pptx|application/vnd.openxmlformats-officedocument.presentationml.presentation|

|.py|text/x-python|

|.py|text/x-script.python|

|.rb|text/x-ruby|

|.sh|application/x-sh|

|.tex|text/x-tex|

|.ts|application/typescript|

|.txt|text/plain|



Usage notes

-----------



||

|ResponsesChat CompletionsAssistants|Tier 1100 RPMTier 2 and 3500 RPMTier 4 and 51000 RPM|PricingZDR and data residency|



Was this page useful?

