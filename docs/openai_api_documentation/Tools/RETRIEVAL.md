Retrieval

=========



Search your data using semantic similarity.



The \*\*Retrieval API\*\* allows you to perform \[\*\*semantic search\*\*](/docs/guides/retrieval#semantic-search) over your data, which is a technique that surfaces semantically similar results — even when they match few or no keywords. Retrieval is useful on its own, but is especially powerful when combined with our models to synthesize responses.



!\[Retrieval depiction](https://cdn.openai.com/API/docs/images/retrieval-depiction.png)



The Retrieval API is powered by \[\*\*vector stores\*\*](/docs/guides/retrieval#vector-stores), which serve as indices for your data. This guide will cover how to perform semantic search, and go into the details of vector stores.



Quickstart

----------



\*   \*\*Create vector store\*\* and upload files.

&nbsp;   



Create vector store with files



```

from openai import OpenAI

client = OpenAI()



vector\_store = client.vector\_stores.create(        # Create vector store

&nbsp;   name="Support FAQ",

)



client.vector\_stores.files.upload\_and\_poll(        # Upload file

&nbsp;   vector\_store\_id=vector\_store.id,

&nbsp;   file=open("customer\_policies.txt", "rb")

)

```



```

import OpenAI from "openai";

const client = new OpenAI();



const vector\_store = await client.vectorStores.create({   // Create vector store

&nbsp;   name: "Support FAQ",

});



await client.vector\_stores.files.upload\_and\_poll({         // Upload file

&nbsp;   vector\_store\_id: vector\_store.id,

&nbsp;   file: fs.createReadStream("customer\_policies.txt"),

});

```



\*   \*\*Send search query\*\* to get relevant results.

&nbsp;   



Search query



```

user\_query = "What is the return policy?"



results = client.vector\_stores.search(

&nbsp;   vector\_store\_id=vector\_store.id,

&nbsp;   query=user\_query,

)

```



```

const userQuery = "What is the return policy?";



const results = await client.vectorStores.search({

&nbsp;   vector\_store\_id: vector\_store.id,

&nbsp;   query: userQuery,

});

```



To learn how to use the results with our models, check out the \[synthesizing responses](/docs/guides/retrieval#synthesizing-responses) section.



Semantic search

---------------



\*\*Semantic search\*\* is a technique that leverages \[vector embeddings](/docs/guides/embeddings) to surface semantically relevant results. Importantly, this includes results with few or no shared keywords, which classical search techniques might miss.



For example, let's look at potential results for `"When did we go to the moon?"`:



|Text|Keyword Similarity|Semantic Similarity|

|---|---|---|

|The first lunar landing occurred in July of 1969.|0%|65%|

|The first man on the moon was Neil Armstrong.|27%|43%|

|When I ate the moon cake, it was delicious.|40%|28%|



\_(\[Jaccard](https://en.wikipedia.org/wiki/Jaccard\_index) used for keyword, \[cosine](https://en.wikipedia.org/wiki/Cosine\_similarity) with `text-embedding-3-small` used for semantic.)\_



Notice how the most relevant result contains none of the words in the search query. This flexibility makes semantic search a very powerful technique for querying knowledge bases of any size.



Semantic search is powered by \[vector stores](/docs/guides/retrieval#vector-stores), which we cover in detail later in the guide. This section will focus on the mechanics of semantic search.



\### Performing semantic search



You can query a vector store using the `search` function and specifying a `query` in natural language. This will return a list of results, each with the relevant chunks, similarity scores, and file of origin.



Search query



```

results = client.vector\_stores.search(

&nbsp;   vector\_store\_id=vector\_store.id,

&nbsp;   query="How many woodchucks are allowed per passenger?",

)

```



```

const results = await client.vectorStores.search({

&nbsp;   vector\_store\_id: vector\_store.id,

&nbsp;   query: "How many woodchucks are allowed per passenger?",

});

```



Results



```

{

&nbsp; "object": "vector\_store.search\_results.page",

&nbsp; "search\_query": "How many woodchucks are allowed per passenger?",

&nbsp; "data": \[

&nbsp;   {

&nbsp;     "file\_id": "file-12345",

&nbsp;     "filename": "woodchuck\_policy.txt",

&nbsp;     "score": 0.85,

&nbsp;     "attributes": {

&nbsp;       "region": "North America",

&nbsp;       "author": "Wildlife Department"

&nbsp;     },

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "text",

&nbsp;         "text": "According to the latest regulations, each passenger is allowed to carry up to two woodchucks."

&nbsp;       },

&nbsp;       {

&nbsp;         "type": "text",

&nbsp;         "text": "Ensure that the woodchucks are properly contained during transport."

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   {

&nbsp;     "file\_id": "file-67890",

&nbsp;     "filename": "transport\_guidelines.txt",

&nbsp;     "score": 0.75,

&nbsp;     "attributes": {

&nbsp;       "region": "North America",

&nbsp;       "author": "Transport Authority"

&nbsp;     },

&nbsp;     "content": \[

&nbsp;       {

&nbsp;         "type": "text",

&nbsp;         "text": "Passengers must adhere to the guidelines set forth by the Transport Authority regarding the transport of woodchucks."

&nbsp;       }

&nbsp;     ]

&nbsp;   }

&nbsp; ],

&nbsp; "has\_more": false,

&nbsp; "next\_page": null

}

```



A response will contain 10 results maximum by default, but you can set up to 50 using the `max\_num\_results` param.



\### Query rewriting



Certain query styles yield better results, so we've provided a setting to automatically rewrite your queries for optimal performance. Enable this feature by setting `rewrite\_query=true` when performing a `search`.



The rewritten query will be available in the result's `search\_query` field.



|Original|Rewritten|

|---|---|

|I'd like to know the height of the main office building.|primary office building height|

|What are the safety regulations for transporting hazardous materials?|safety regulations for hazardous materials|

|How do I file a complaint about a service issue?|service complaint filing process|



\### Attribute filtering



Attribute filtering helps narrow down results by applying criteria, such as restricting searches to a specific date range. You can define and combine criteria in `attribute\_filter` to target files based on their attributes before performing semantic search.



Use \*\*comparison filters\*\* to compare a specific `key` in a file's `attributes` with a given `value`, and \*\*compound filters\*\* to combine multiple filters using `and` and `or`.



Comparison filter



```

{

&nbsp; "type": "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "in" | "nin",  // comparison operators

&nbsp; "key": "attributes\_key",                           // attributes key

&nbsp; "value": "target\_value"                             // value to compare against

}

```



Compound filter



```

{

&nbsp; "type": "and" | "or",                                // logical operators

&nbsp; "filters": \[...]                                   

}

```



Below are some example filters.



Region



Filter for a region



```

{

&nbsp; "type": "eq",

&nbsp; "key": "region",

&nbsp; "value": "us"

}

```



Date range



Filter for a date range



```

{

&nbsp; "type": "and",

&nbsp; "filters": \[

&nbsp;   {

&nbsp;     "type": "gte",

&nbsp;     "key": "date",

&nbsp;     "value": 1704067200  // unix timestamp for 2024-01-01

&nbsp;   },

&nbsp;   {

&nbsp;     "type": "lte",

&nbsp;     "key": "date",

&nbsp;     "value": 1710892800  // unix timestamp for 2024-03-20

&nbsp;   }

&nbsp; ]

}

```



Filenames



Filter to match any of a set of filenames



```

{

&nbsp; "type": "in",

&nbsp; "property": "filename",

&nbsp; "value": \["example.txt", "example2.txt"]

}

```



Exclude filenames



Filter to exclude drafts by filename



```

{

&nbsp; "type": "nin",

&nbsp; "property": "filename",

&nbsp; "value": \["draft.txt", "internal\_notes.md"]

}

```



Complex



Filter for top secret projects with certain names in english



```

{

&nbsp; "type": "or",

&nbsp; "filters": \[

&nbsp;   {

&nbsp;     "type": "and",

&nbsp;     "filters": \[

&nbsp;       {

&nbsp;         "type": "or",

&nbsp;         "filters": \[

&nbsp;           {

&nbsp;             "type": "eq",

&nbsp;             "key": "project\_code",

&nbsp;             "value": "X123"

&nbsp;           },

&nbsp;           {

&nbsp;             "type": "eq",

&nbsp;             "key": "project\_code",

&nbsp;             "value": "X999"

&nbsp;           }

&nbsp;         ]

&nbsp;       },

&nbsp;       {

&nbsp;         "type": "eq",

&nbsp;         "key": "confidentiality",

&nbsp;         "value": "top\_secret"

&nbsp;       }

&nbsp;     ]

&nbsp;   },

&nbsp;   {

&nbsp;     "type": "eq",

&nbsp;     "key": "language",

&nbsp;     "value": "en"

&nbsp;   }

&nbsp; ]

}

```



\### Ranking



If you find that your file search results are not sufficiently relevant, you can adjust the `ranking\_options` to improve the quality of responses. This includes specifying a `ranker`, such as `auto` or `default-2024-08-21`, and setting a `score\_threshold` between 0.0 and 1.0. A higher `score\_threshold` will limit the results to more relevant chunks, though it may exclude some potentially useful ones. When `ranking\_options.hybrid\_search` is provided you can also tune `hybrid\_search.embedding\_weight` (`rrf\_embedding\_weight`) and `hybrid\_search.text\_weight` (`rrf\_text\_weight`) to control how reciprocal rank fusion balances semantic embedding matches vs. sparse keyword matches. Increase the former to emphasize semantic similarity, increase the latter to emphasize textual overlap, and ensure at least one of the weights is greater than zero.



Vector stores

-------------



Vector stores are the containers that power semantic search for the Retrieval API and the \[file search](/docs/guides/tools-file-search) tool. When you add a file to a vector store it will be automatically chunked, embedded, and indexed.



Vector stores contain `vector\_store\_file` objects, which are backed by a `file` object.



|Object type|Description|

|---|---|

|file|Represents content uploaded through the Files API. Often used with vector stores, but also for fine-tuning and other use cases.|

|vector\_store|Container for searchable files.|

|vector\_store.file|Wrapper type specifically representing a file that has been chunked and embedded, and has been associated with a vector\_store.Contains attributes map used for filtering.|



\### Pricing



You will be charged based on the total storage used across all your vector stores, determined by the size of parsed chunks and their corresponding embeddings.



|Storage|Cost|

|---|---|

|Up to 1 GB (across all stores)|Free|

|Beyond 1 GB|$0.10/GB/day|



See \[expiration policies](/docs/guides/retrieval#expiration-policies) for options to minimize costs.



\### Vector store operations



Create



Create vector store



```

client.vector\_stores.create(

&nbsp;   name="Support FAQ",

&nbsp;   file\_ids=\["file\_123"]

)

```



```

await client.vector\_stores.create({

&nbsp;   name: "Support FAQ",

&nbsp;   file\_ids: \["file\_123"]

});

```



Retrieve



Retrieve vector store



```

client.vector\_stores.retrieve(

&nbsp;   vector\_store\_id="vs\_123"

)

```



```

await client.vector\_stores.retrieve({

&nbsp;   vector\_store\_id: "vs\_123"

});

```



Update



Update vector store



```

client.vector\_stores.update(

&nbsp;   vector\_store\_id="vs\_123",

&nbsp;   name="Support FAQ Updated"

)

```



```

await client.vector\_stores.update({

&nbsp;   vector\_store\_id: "vs\_123",

&nbsp;   name: "Support FAQ Updated"

});

```



Delete



Delete vector store



```

client.vector\_stores.delete(

&nbsp;   vector\_store\_id="vs\_123"

)

```



```

await client.vector\_stores.delete({

&nbsp;   vector\_store\_id: "vs\_123"

});

```



List



List vector stores



```

client.vector\_stores.list()

```



```

await client.vector\_stores.list();

```



\### Vector store file operations



Some operations, like `create` for `vector\_store.file`, are asynchronous and may take time to complete — use our helper functions, like `create\_and\_poll` to block until it is. Otherwise, you may check the status. Removing files from a vector store is eventually consistent, and search results may still include content from a removed file for a short period.



Create



Create vector store file



```

client.vector\_stores.files.create\_and\_poll(

&nbsp;   vector\_store\_id="vs\_123",

&nbsp;   file\_id="file\_123"

)

```



```

await client.vector\_stores.files.create\_and\_poll({

&nbsp;   vector\_store\_id: "vs\_123",

&nbsp;   file\_id: "file\_123"

});

```



Upload



Upload vector store file



```

client.vector\_stores.files.upload\_and\_poll(

&nbsp;   vector\_store\_id="vs\_123",

&nbsp;   file=open("customer\_policies.txt", "rb")

)

```



```

await client.vector\_stores.files.upload\_and\_poll({

&nbsp;   vector\_store\_id: "vs\_123",

&nbsp;   file: fs.createReadStream("customer\_policies.txt"),

});

```



Retrieve



Retrieve vector store file



```

client.vector\_stores.files.retrieve(

&nbsp;   vector\_store\_id="vs\_123",

&nbsp;   file\_id="file\_123"

)

```



```

await client.vector\_stores.files.retrieve({

&nbsp;   vector\_store\_id: "vs\_123",

&nbsp;   file\_id: "file\_123"

});

```



Update



Update vector store file



```

client.vector\_stores.files.update(

&nbsp;   vector\_store\_id="vs\_123",

&nbsp;   file\_id="file\_123",

&nbsp;   attributes={"key": "value"}

)

```



```

await client.vector\_stores.files.update({

&nbsp;   vector\_store\_id: "vs\_123",

&nbsp;   file\_id: "file\_123",

&nbsp;   attributes: { key: "value" }

});

```



Delete



Delete vector store file



```

client.vector\_stores.files.delete(

&nbsp;   vector\_store\_id="vs\_123",

&nbsp;   file\_id="file\_123"

)

```



```

await client.vector\_stores.files.delete({

&nbsp;   vector\_store\_id: "vs\_123",

&nbsp;   file\_id: "file\_123"

});

```



List



List vector store files



```

client.vector\_stores.files.list(

&nbsp;   vector\_store\_id="vs\_123"

)

```



```

await client.vector\_stores.files.list({

&nbsp;   vector\_store\_id: "vs\_123"

});

```



\### Batch operations



Create



Batch create operation



```

client.vector\_stores.file\_batches.create\_and\_poll(

&nbsp;   vector\_store\_id="vs\_123",

&nbsp;   files=\[

&nbsp;       {

&nbsp;           "file\_id": "file\_123",

&nbsp;           "attributes": {"department": "finance"}

&nbsp;       },

&nbsp;       {

&nbsp;           "file\_id": "file\_456",

&nbsp;           "chunking\_strategy": {

&nbsp;               "type": "static",

&nbsp;               "max\_chunk\_size\_tokens": 1200,

&nbsp;               "chunk\_overlap\_tokens": 200

&nbsp;           }

&nbsp;       }

&nbsp;   ]

)

```



```

await client.vector\_stores.file\_batches.create\_and\_poll({

&nbsp;   vector\_store\_id: "vs\_123",

&nbsp;   files: \[

&nbsp;       {

&nbsp;           file\_id: "file\_123",

&nbsp;           attributes: { department: "finance" }

&nbsp;       },

&nbsp;       {

&nbsp;           file\_id: "file\_456",

&nbsp;           chunking\_strategy: {

&nbsp;               type: "static",

&nbsp;               max\_chunk\_size\_tokens: 1200,

&nbsp;               chunk\_overlap\_tokens: 200

&nbsp;           }

&nbsp;       }

&nbsp;   ]

});

```



Retrieve



Batch retrieve operation



```

client.vector\_stores.file\_batches.retrieve(

&nbsp;   vector\_store\_id="vs\_123",

&nbsp;   batch\_id="vsfb\_123"

)

```



```

await client.vector\_stores.file\_batches.retrieve({

&nbsp;   vector\_store\_id: "vs\_123",

&nbsp;   batch\_id: "vsfb\_123"

});

```



Cancel



Batch cancel operation



```

client.vector\_stores.file\_batches.cancel(

&nbsp;   vector\_store\_id="vs\_123",

&nbsp;   batch\_id="vsfb\_123"

)

```



```

await client.vector\_stores.file\_batches.cancel({

&nbsp;   vector\_store\_id: "vs\_123",

&nbsp;   batch\_id: "vsfb\_123"

});

```



List



Batch list operation



```

client.vector\_stores.file\_batches.list(

&nbsp;   vector\_store\_id="vs\_123"

)

```



```

await client.vector\_stores.file\_batches.list({

&nbsp;   vector\_store\_id: "vs\_123"

});

```



When creating a batch you can either provide `file\_ids` with optional `attributes` and/or `chunking\_strategy`, or use the `files` array to pass objects that include a `file\_id` plus optional `attributes` and `chunking\_strategy` for each file. The two options are mutually exclusive so that you can cleanly control whether every file shares the same settings or you need per-file overrides.



\### Attributes



Each `vector\_store.file` can have associated `attributes`, a dictionary of values that can be referenced when performing \[semantic search](/docs/guides/retrieval#semantic-search) with \[attribute filtering](/docs/guides/retrieval#attribute-filtering). The dictionary can have at most 16 keys, with a limit of 256 characters each.



Create vector store file with attributes



```

client.vector\_stores.files.create(

&nbsp;   vector\_store\_id="<vector\_store\_id>",

&nbsp;   file\_id="file\_123",

&nbsp;   attributes={

&nbsp;       "region": "US",

&nbsp;       "category": "Marketing",

&nbsp;       "date": 1672531200      # Jan 1, 2023

&nbsp;   }

)

```



```

await client.vector\_stores.files.create(<vector\_store\_id>, {

&nbsp;   file\_id: "file\_123",

&nbsp;   attributes: {

&nbsp;       region: "US",

&nbsp;       category: "Marketing",

&nbsp;       date: 1672531200, // Jan 1, 2023

&nbsp;   },

});

```



\### Expiration policies



You can set an expiration policy on `vector\_store` objects with `expires\_after`. Once a vector store expires, all associated `vector\_store.file` objects will be deleted and you'll no longer be charged for them.



Set expiration policy for vector store



```

client.vector\_stores.update(

&nbsp;   vector\_store\_id="vs\_123",

&nbsp;   expires\_after={

&nbsp;       "anchor": "last\_active\_at",

&nbsp;       "days": 7

&nbsp;   }

)

```



```

await client.vector\_stores.update({

&nbsp;   vector\_store\_id: "vs\_123",

&nbsp;   expires\_after: {

&nbsp;       anchor: "last\_active\_at",

&nbsp;       days: 7,

&nbsp;   },

});

```



\### Limits



The maximum file size is 512 MB. Each file should contain no more than 5,000,000 tokens per file (computed automatically when you attach a file).



\### Chunking



By default, `max\_chunk\_size\_tokens` is set to `800` and `chunk\_overlap\_tokens` is set to `400`, meaning every file is indexed by being split up into 800-token chunks, with 400-token overlap between consecutive chunks.



You can adjust this by setting \[`chunking\_strategy`](/docs/api-reference/vector-stores-files/createFile#vector-stores-files-createfile-chunking\_strategy) when adding files to the vector store. There are certain limitations to `chunking\_strategy`:



\*   `max\_chunk\_size\_tokens` must be between 100 and 4096 inclusive.

\*   `chunk\_overlap\_tokens` must be non-negative and should not exceed `max\_chunk\_size\_tokens / 2`.



Supported file types



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



Synthesizing responses

----------------------



After performing a query you may want to synthesize a response based on the results. You can leverage our models to do so, by supplying the results and original query, to get back a grounded response.



Perform search query to get results



```

from openai import OpenAI



client = OpenAI()



user\_query = "What is the return policy?"



results = client.vector\_stores.search(

&nbsp;   vector\_store\_id=vector\_store.id,

&nbsp;   query=user\_query,

)

```



```

import OpenAI from "openai";

const client = new OpenAI();



const userQuery = "What is the return policy?";



const results = await client.vectorStores.search({

&nbsp;   vector\_store\_id: vector\_store.id,

&nbsp;   query: userQuery,

});

```



Synthesize a response based on results



```

formatted\_results = format\_results(results.data)



'\\n'.join('\\n'.join(c.text) for c in result.content for result in results.data)



completion = client.chat.completions.create(

&nbsp;   model="gpt-4.1",

&nbsp;   messages=\[

&nbsp;       {

&nbsp;           "role": "developer",

&nbsp;           "content": "Produce a concise answer to the query based on the provided sources."

&nbsp;       },

&nbsp;       {

&nbsp;           "role": "user",

&nbsp;           "content": f"Sources: {formatted\_results}\\n\\nQuery: '{user\_query}'"

&nbsp;       }

&nbsp;   ],

)



print(completion.choices\[0].message.content)

```



```

const formattedResults = formatResults(results.data);

// Join the text content of all results

const textSources = results.data.map(result => result.content.map(c => c.text).join('\\n')).join('\\n');



const completion = await client.chat.completions.create({

&nbsp;   model: "gpt-4.1",

&nbsp;   messages: \[

&nbsp;       {

&nbsp;           role: "developer",

&nbsp;           content: "Produce a concise answer to the query based on the provided sources."

&nbsp;       },

&nbsp;       {

&nbsp;           role: "user",

&nbsp;           content: `Sources: ${formattedResults}\\n\\nQuery: '${userQuery}'`

&nbsp;       }

&nbsp;   ],

});



console.log(completion.choices\[0].message.content);

```



```

"Our return policy allows returns within 30 days of purchase."

```



This uses a sample `format\_results` function, which could be implemented like so:



Sample result formatting function



```

def format\_results(results):

&nbsp;   formatted\_results = ''

&nbsp;   for result in results.data:

&nbsp;       formatted\_result = f"<result file\_id='{result.file\_id}' file\_name='{result.file\_name}'>"

&nbsp;       for part in result.content:

&nbsp;           formatted\_result += f"<content>{part.text}</content>"

&nbsp;       formatted\_results += formatted\_result + "</result>"

&nbsp;   return f"<sources>{formatted\_results}</sources>"

```



```

function formatResults(results) {

&nbsp;   let formattedResults = '';

&nbsp;   for (const result of results.data) {

&nbsp;       let formattedResult = `<result file\_id='${result.file\_id}' file\_name='${result.file\_name}'>`;

&nbsp;       for (const part of result.content) {

&nbsp;           formattedResult += `<content>${part.text}</content>`;

&nbsp;       }

&nbsp;       formattedResults += formattedResult + "</result>";

&nbsp;   }

&nbsp;   return `<sources>${formattedResults}</sources>`;

}

```



Was this page useful?

