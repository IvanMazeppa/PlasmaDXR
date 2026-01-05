Web search

==========



Allow models to search the web for the latest information before generating a response.



Web search allows models to access up-to-date information from the internet and provide answers with sourced citations. To enable this, use the web search tool in the Responses API or, in some cases, Chat Completions.



There are three main types of web search available with OpenAI models:



1\.  Non‑reasoning web search: The non-reasoning model sends the user’s query to the web search tool, which returns the response based on top results. There’s no internal planning and the model simply passes along the search tool’s responses. This method is fast and ideal for quick lookups.

2\.  Agentic search with reasoning models is an approach where the model actively manages the search process. It can perform web searches as part of its chain of thought, analyze results, and decide whether to keep searching. This flexibility makes agentic search well suited to complex workflows, but it also means searches take longer than quick lookups. For example, you can adjust GPT-5’s reasoning level to change both the depth and latency of the search.

3\.  Deep research is a specialized, agent-driven method for in-depth, extended investigations by reasoning models. The model conducts web searches as part of its chain of thought, often tapping into hundreds of sources. Deep research can run for several minutes and is best used with background mode. These tasks typically use models like `o3-deep-research`, `o4-mini-deep-research`, or `gpt-5` with reasoning level set to `high`.



Using the \[Responses API](/docs/api-reference/responses), you can enable web search by configuring it in the `tools` array in an API request to generate content. Like any other tool, the model can choose to search the web or not based on the content of the input prompt.



Web search tool example



```

import OpenAI from "openai";

const client = new OpenAI();



const response = await client.responses.create({

&nbsp;   model: "gpt-5",

&nbsp;   tools: \[

&nbsp;       { type: "web\_search" },

&nbsp;   ],

&nbsp;   input: "What was a positive news story from today?",

});



console.log(response.output\_text);

```



```

from openai import OpenAI

client = OpenAI()



response = client.responses.create(

&nbsp;   model="gpt-5",

&nbsp;   tools=\[{"type": "web\_search"}],

&nbsp;   input="What was a positive news story from today?"

)



print(response.output\_text)

```



```

curl "https://api.openai.com/v1/responses" \\

&nbsp;   -H "Content-Type: application/json" \\

&nbsp;   -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp;   -d '{

&nbsp;       "model": "gpt-5",

&nbsp;       "tools": \[{"type": "web\_search"}],

&nbsp;       "input": "what was a positive news story from today?"

&nbsp;   }'

```



```

using OpenAI.Responses;



string key = Environment.GetEnvironmentVariable("OPENAI\_API\_KEY")!;

OpenAIResponseClient client = new(model: "gpt-5", apiKey: key);



ResponseCreationOptions options = new();

options.Tools.Add(ResponseTool.CreateWebSearchTool());



OpenAIResponse response = (OpenAIResponse)client.CreateResponse(\[

&nbsp;   ResponseItem.CreateUserMessageItem(\[

&nbsp;       ResponseContentPart.CreateInputTextPart("What was a positive news story from today?"),

&nbsp;   ]),

], options);



Console.WriteLine(response.GetOutputText());

```



Output and citations

--------------------



Model responses that use the web search tool will include two parts:



\*   A `web\_search\_call` output item with the ID of the search call, along with the action taken in `web\_search\_call.action`. The action is one of:

&nbsp;   \*   `search`, which represents a web search. It will usually (but not always) includes the search `query` and `domains` which were searched. Search actions incur a tool call cost (see \[pricing](/docs/pricing#built-in-tools)).

&nbsp;   \*   `open\_page`, which represents a page being opened. Supported in reasoning models.

&nbsp;   \*   `find\_in\_page`, which represents searching within a page. Supported in reasoning models.

\*   A `message` output item containing:

&nbsp;   \*   The text result in `message.content\[0].text`

&nbsp;   \*   Annotations `message.content\[0].annotations` for the cited URLs



By default, the model's response will include inline citations for URLs found in the web search results. In addition to this, the `url\_citation` annotation object will contain the URL, title and location of the cited source.



When displaying web results or information contained in web results to end users, inline citations must be made clearly visible and clickable in your user interface.



```

\[

&nbsp;   {

&nbsp;       "type": "web\_search\_call",

&nbsp;       "id": "ws\_67c9fa0502748190b7dd390736892e100be649c1a5ff9609",

&nbsp;       "status": "completed"

&nbsp;   },

&nbsp;   {

&nbsp;       "id": "msg\_67c9fa077e288190af08fdffda2e34f20be649c1a5ff9609",

&nbsp;       "type": "message",

&nbsp;       "status": "completed",

&nbsp;       "role": "assistant",

&nbsp;       "content": \[

&nbsp;           {

&nbsp;               "type": "output\_text",

&nbsp;               "text": "On March 6, 2025, several news...",

&nbsp;               "annotations": \[

&nbsp;                   {

&nbsp;                       "type": "url\_citation",

&nbsp;                       "start\_index": 2606,

&nbsp;                       "end\_index": 2758,

&nbsp;                       "url": "https://...",

&nbsp;                       "title": "Title..."

&nbsp;                   }

&nbsp;               ]

&nbsp;           }

&nbsp;       ]

&nbsp;   }

]

```



Domain filtering

----------------



Domain filtering in web search lets you limit results to a specific set of domains. With the `filters` parameter you can set an allow-list of up to 100 URLs. When formatting URLs, omit the HTTP or HTTPS prefix. For example, use `openai.com` instead of `https://openai.com/`. This approach also includes subdomains in the search. Note that domain filtering is only available in the Responses API with the `web\_search` tool.



Sources

-------



To view all URLs retrieved during a web search, use the `sources` field. Unlike inline citations, which show only the most relevant references, sources returns the complete list of URLs the model consulted when forming its response. The number of sources is often greater than the number of citations. Real-time third-party feeds are also surfaced here and are labeled as `oai-sports`, `oai-weather`, or `oai-finance`. The sources field is available with both the `web\_search` and `web\_search\_preview` tools.



List sources



```

curl "https://api.openai.com/v1/responses" \\

-H "Content-Type: application/json" \\

-H "Authorization: Bearer $OPENAI\_API\_KEY" \\

-d '{

&nbsp; "model": "gpt-5",

&nbsp; "reasoning": { "effort": "low" },

&nbsp; "tools": \[

&nbsp;   {

&nbsp;     "type": "web\_search",

&nbsp;     "filters": {

&nbsp;       "allowed\_domains": \[

&nbsp;         "pubmed.ncbi.nlm.nih.gov",

&nbsp;         "clinicaltrials.gov",

&nbsp;         "www.who.int",

&nbsp;         "www.cdc.gov",

&nbsp;         "www.fda.gov"

&nbsp;       ]

&nbsp;     }

&nbsp;   }

&nbsp; ],

&nbsp; "tool\_choice": "auto",

&nbsp; "include": \["web\_search\_call.action.sources"],

&nbsp; "input": "Please perform a web search on how semaglutide is used in the treatment of diabetes."

}'

```



```

import OpenAI from "openai";

const client = new OpenAI();



const response = await client.responses.create({

&nbsp; model: "gpt-5",

&nbsp; reasoning: { effort: "low" },

&nbsp; tools: \[

&nbsp;     {

&nbsp;         type: "web\_search",

&nbsp;         filters: {

&nbsp;             allowed\_domains: \[

&nbsp;                 "pubmed.ncbi.nlm.nih.gov",

&nbsp;                 "clinicaltrials.gov",

&nbsp;                 "www.who.int",

&nbsp;                 "www.cdc.gov",

&nbsp;                 "www.fda.gov",

&nbsp;             ],

&nbsp;         },

&nbsp;     },

&nbsp; ],

&nbsp; tool\_choice: "auto",

&nbsp; include: \["web\_search\_call.action.sources"],

&nbsp; input: "Please perform a web search on how semaglutide is used in the treatment of diabetes.",

});



console.log(response.output\_text);

```



```

from openai import OpenAI

client = OpenAI()



response = client.responses.create(

&nbsp; model="gpt-5",

&nbsp; reasoning={"effort": "low"},

&nbsp; tools=\[

&nbsp;     {

&nbsp;         "type": "web\_search",

&nbsp;         "filters": {

&nbsp;             "allowed\_domains": \[

&nbsp;                 "pubmed.ncbi.nlm.nih.gov",

&nbsp;                 "clinicaltrials.gov",

&nbsp;                 "www.who.int",

&nbsp;                 "www.cdc.gov",

&nbsp;                 "www.fda.gov",

&nbsp;             ]

&nbsp;         },

&nbsp;     }

&nbsp; ],

&nbsp; tool\_choice="auto",

&nbsp; include=\["web\_search\_call.action.sources"],

&nbsp; input="Please perform a web search on how semaglutide is used in the treatment of diabetes.",

)



print(response.output\_text)

```



User location

-------------



To refine search results based on geography, you can specify an approximate user location using country, city, region, and/or timezone.



\*   The `city` and `region` fields are free text strings, like `Minneapolis` and `Minnesota` respectively.

\*   The `country` field is a two-letter \[ISO country code](https://en.wikipedia.org/wiki/ISO\_3166-1), like `US`.

\*   The `timezone` field is an \[IANA timezone](https://timeapi.io/documentation/iana-timezones) like `America/Chicago`.



Note that user location is not supported for deep research models using web search.



Customizing user location



```

from openai import OpenAI

client = OpenAI()



response = client.responses.create(

&nbsp;   model="o4-mini",

&nbsp;   tools=\[{

&nbsp;       "type": "web\_search",

&nbsp;       "user\_location": {

&nbsp;           "type": "approximate",

&nbsp;           "country": "GB",

&nbsp;           "city": "London",

&nbsp;           "region": "London",

&nbsp;       }

&nbsp;   }],

&nbsp;   input="What are the best restaurants near me?",

)



print(response.output\_text)

```



```

using OpenAI.Responses;



string key = Environment.GetEnvironmentVariable("OPENAI\_API\_KEY")!;

OpenAIResponseClient client = new(model: "gpt-5", apiKey: key);



ResponseCreationOptions options = new();

options.Tools.Add(ResponseTool.CreateWebSearchTool(

&nbsp;   userLocation: WebSearchToolLocation.CreateApproximateLocation(

&nbsp;       country: "GB",

&nbsp;       city: "London",

&nbsp;       region: "Granary Square"

&nbsp;   )

));



OpenAIResponse response = (OpenAIResponse)client.CreateResponse(\[

&nbsp;   ResponseItem.CreateUserMessageItem(\[

&nbsp;       ResponseContentPart.CreateInputTextPart(

&nbsp;           "What are the best restaurants near me?"

&nbsp;       )

&nbsp;   ])

], options);



Console.WriteLine(response.GetOutputText());

```



```

import OpenAI from "openai";

const openai = new OpenAI();



const response = await openai.responses.create({

&nbsp;   model: "o4-mini",

&nbsp;   tools: \[{

&nbsp;       type: "web\_search",

&nbsp;       user\_location: {

&nbsp;           type: "approximate",

&nbsp;           country: "GB",

&nbsp;           city: "London",

&nbsp;           region: "London"

&nbsp;       }

&nbsp;   }],

&nbsp;   input: "What are the best restaurants near me?",

});

console.log(response.output\_text);

```



```

curl "https://api.openai.com/v1/responses" \\

&nbsp;   -H "Content-Type: application/json" \\

&nbsp;   -H "Authorization: Bearer $OPENAI\_API\_KEY" \\

&nbsp;   -d '{

&nbsp;       "model": "o4-mini",

&nbsp;       "tools": \[{

&nbsp;           "type": "web\_search",

&nbsp;           "user\_location": {

&nbsp;               "type": "approximate",

&nbsp;               "country": "GB",

&nbsp;               "city": "London",

&nbsp;               "region": "London"

&nbsp;           }

&nbsp;       }],

&nbsp;       "input": "What are the best restaurants near me?"

&nbsp;   }'

```



Live internet access

--------------------



Control whether the web search tool fetches live content or uses only cached/indexed results in the Responses API.



\*   Set `external\_web\_access: false` on the `web\_search` tool to run in offline/cache‑only mode.

\*   Default is `true` (live access) if you do not set it.

\*   Preview variants (`web\_search\_preview`) ignore this parameter and behave as if `external\_web\_access` is `true`.



Control live internet access



```

curl "https://api.openai.com/v1/responses" -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI\_API\_KEY" -d '{

&nbsp; "model": "gpt-5",

&nbsp; "tools": \[

&nbsp;   { "type": "web\_search", "external\_web\_access": false }

&nbsp; ],

&nbsp; "tool\_choice": "auto",

&nbsp; "input": "Find the sunrise time in Paris today and cite the source."

}'

```



```

import OpenAI from "openai";

const client = new OpenAI();



const response = await client.responses.create({

model: "gpt-5",

tools: \[

&nbsp; { type: "web\_search", external\_web\_access: false },

],

tool\_choice: "auto",

input: "Find the sunrise time in Paris today and cite the source.",

});



console.log(response.output\_text);

```



```

from openai import OpenAI

client = OpenAI()



resp = client.responses.create(

&nbsp; model="gpt-5",

&nbsp; tools=\[{ "type": "web\_search", "external\_web\_access": False }],

&nbsp; tool\_choice="auto",

&nbsp; input="Find the sunrise time in Paris today and cite the source.",

)

print(resp.output\_text)

```



API compatibility

-----------------



Web search is available in the Responses API as the generally available version of the tool, `web\_search`, as well as the earlier tool version, `web\_search\_preview`. To use web search in the Chat Completions API, use the specialized web search models `gpt-5-search-api`, `gpt-4o-search-preview` and `gpt-4o-mini-search-preview`.



Limitations

-----------



\*   Web search is currently not supported in \[`gpt-5`](/docs/models/gpt-5) with `minimal` reasoning, and \[`gpt-4.1-nano`](/docs/models/gpt-4.1-nano).

\*   When used as a tool in the \[Responses API](/docs/api-reference/responses), web search has the same tiered rate limits as the models above.

\*   Web search is limited to a context window size of 128000 (even with \[`gpt-4.1`](/docs/models/gpt-4.1) and \[`gpt-4.1-mini`](/docs/models/gpt-4.1-mini) models).



Usage notes

-----------



||

|ResponsesChat CompletionsAssistants|Same as tiered rate limits for underlying model used with the tool.|PricingZDR and data residency|



Was this page useful?

